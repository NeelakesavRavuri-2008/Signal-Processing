"""
Streamlit real-time dashboard for RF interference detection.
Run:  streamlit run dashboard.py

Full pipeline (Phase 1-13):
  Phase 1-2:  IQ load, PSD, CA-CFAR, event extraction
  Phase 3:    Pd / Pfa evaluation
  Phase 4-5:  12-D features + Random Forest classifier
  Phase 6:    Kalman tracking
  Phase 7:    This dashboard
  Phase 8:    OS-CFAR / VI-CFAR (mode selectable)
  Phase 9:    Blackman-Harris window + morphological cleanup
  Phase 10:   XGBoost + Isolation Forest (unknown jammer detection)
  Phase 11:   RTL-SDR live data source
  Phase 12:   SQLite event logging + CCI alerts
  Phase 13:   1D-CNN deep learning classifier
"""
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from rf_pipeline import (
    # Config
    FS, N_FFT, CFAR_OFFSET_DB,
    GROUND_TRUTH_JAMMERS_MHZ,
    PROTECTED_CENTER, PROTECTED_BANDWIDTH,
    XGB_AVAILABLE, SDR_AVAILABLE,
    # Phase 1-2
    simulate_frame, compute_linear_psd, linear_ca_cfar, extract_events,
    # Phase 3
    evaluate_pd_pfa,
    # Phase 5
    classify_events, load_classifier, train_classifier, build_event_dataset,
    # Phase 6
    track_frames,
    # Phase 8
    os_cfar, vi_cfar,
    # Phase 9
    compute_linear_psd_v2, clean_detection_mask,
    # Phase 10
    load_xgboost, train_xgboost, classify_events_xgb,
    # Phase 11
    RTLSDRIQSource,
    # Phase 12
    init_event_database, log_events_to_db,
    query_recent_events, check_protected_band_intrusion,
)

# Phase 13: optional CNN import
try:
    from cnn_classifier import (
        CNN_AVAILABLE, load_cnn, train_cnn,
        build_cnn_dataset, classify_psd_cnn,
    )
except ImportError:
    CNN_AVAILABLE = False


# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(page_title="RF Interference Monitor",
                   layout="wide", page_icon="📡")
st.title("📡 Real-Time RF Interference Detection & Classification")


# ==============================================================================
# SIDEBAR - CONFIGURATION
# ==============================================================================
st.sidebar.header("Configuration")

update_rate = st.sidebar.slider("Update rate (s)", 0.05, 2.0, 0.2, 0.05,
                                help="Lower = faster refresh, more CPU")
cfar_offset = st.sidebar.slider("CFAR offset (dB)", 5.0, 25.0, 15.0, 0.5)
jam_amp = st.sidebar.slider("Jammer amplitude", 0.1, 1.5, 1.0, 0.05)
n_jammers = st.sidebar.slider("Number of jammers", 1, 10, 5, 1,
                              help="More jammers = busier dashboard, richer tracking")
n_display = st.sidebar.slider("Frames in waterfall", 20, 500, 150, 10,
                              help="Higher = longer history in track plot")

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Settings")

# --- Phase 8: CFAR Mode ---
cfar_mode = st.sidebar.radio(
    "CFAR Mode",
    ["CA-CFAR", "OS-CFAR", "VI-CFAR"],
    index=0,
    help="CA = Cell Averaging, OS = Ordered Statistic, VI = Variability Index"
)

# --- Phase 9: PSD window + Morphology ---
use_bh = st.sidebar.checkbox(
    "Blackman-Harris window", value=True,
    help="Lower side-lobes than Hann (-92 dB vs -31 dB)"
)
use_morph = st.sidebar.checkbox(
    "Morphological cleanup", value=True,
    help="Removes isolated false detections"
)

# --- Phase 5 / 10 / 13: Classifier ---
classifier_options = ["Random Forest"]
if XGB_AVAILABLE:
    classifier_options.append("XGBoost")
if CNN_AVAILABLE:
    classifier_options.append("1D-CNN (deep learning)")

classifier_choice = st.sidebar.radio(
    "Classifier", classifier_options, index=0
)

# --- Phase 11: Data Source ---
data_source_options = ["Simulation"]
if SDR_AVAILABLE:
    data_source_options.append("Live RTL-SDR")
data_source = st.sidebar.radio("Data Source", data_source_options, index=0)

# --- Phase 6: Tracking ---
enable_track = st.sidebar.checkbox("Enable Kalman tracking", True)

# --- Phase 12: Logging ---
enable_logging = st.sidebar.checkbox("Log events to SQLite", True)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"XGBoost: {'✅' if XGB_AVAILABLE else '❌'}   "
    f"RTL-SDR: {'✅' if SDR_AVAILABLE else '❌'}   "
    f"CNN: {'✅' if CNN_AVAILABLE else '❌'}"
)


# ==============================================================================
# CLASSIFIER LOADING (cached per classifier type)
# ==============================================================================
@st.cache_resource(show_spinner="Training Random Forest classifier (first run only)...")
def get_rf_classifier():
    clf, sc = load_classifier()
    if clf is None:
        X, y = build_event_dataset(n_per_class=80, seed=7)
        clf, sc = train_classifier(X, y, save=True)
    return clf, sc


@st.cache_resource(show_spinner="Training XGBoost classifier (first run only)...")
def get_xgb_classifier():
    if not XGB_AVAILABLE:
        return None, None, None, None
    clf, sc, le, iso = load_xgboost()
    if clf is None:
        X, y = build_event_dataset(n_per_class=80, seed=7)
        clf, sc, le, iso = train_xgboost(X, y, save=True)
    return clf, sc, le, iso


@st.cache_resource(show_spinner="Training 1D-CNN (first run only, ~60s)...")
def get_cnn_classifier():
    if not CNN_AVAILABLE:
        return None, None
    model, le = load_cnn()
    if model is None:
        X_cnn, y_cnn = build_cnn_dataset(n_per_class=150, seed=7)
        model, le, _ = train_cnn(X_cnn, y_cnn, epochs=15, verbose=0)
    return model, le


# Load the selected classifier
rf_clf = rf_sc = None
xgb_clf = xgb_sc = xgb_le = xgb_iso = None
cnn_model = cnn_le = None

if classifier_choice == "XGBoost" and XGB_AVAILABLE:
    xgb_clf, xgb_sc, xgb_le, xgb_iso = get_xgb_classifier()
elif classifier_choice == "1D-CNN (deep learning)" and CNN_AVAILABLE:
    cnn_model, cnn_le = get_cnn_classifier()
else:
    rf_clf, rf_sc = get_rf_classifier()


# ==============================================================================
# SDR SOURCE INITIALIZATION
# ==============================================================================
if "sdr_source" not in st.session_state:
    if data_source == "Live RTL-SDR" and SDR_AVAILABLE:
        st.session_state.sdr_source = RTLSDRIQSource(
            center_freq=100e6, sample_rate=2.4e6
        )
    else:
        st.session_state.sdr_source = None

sdr = st.session_state.sdr_source


# ==============================================================================
# SESSION STATE
# ==============================================================================
if "frames" not in st.session_state:
    st.session_state.frames = []
if "tracks" not in st.session_state:
    st.session_state.tracks = []
if "pd_list" not in st.session_state:
    st.session_state.pd_list = []
if "pfa_list" not in st.session_state:
    st.session_state.pfa_list = []
if "db_conn" not in st.session_state:
    st.session_state.db_conn = init_event_database()
if "alert_state" not in st.session_state:
    st.session_state.alert_state = False


# ==============================================================================
# LAYOUT PLACEHOLDERS
# ==============================================================================
alert_ph = st.empty()

col1, col2 = st.columns([1.4, 1.0])
place_wf = col1.empty()
place_psd = col2.empty()

m1, m2, m3, m4 = st.columns(4)
kpi_events = m1.empty()
kpi_pd = m2.empty()
kpi_pfa = m3.empty()
kpi_trk = m4.empty()

st.subheader("Detected interference events")
tbl_ph = st.empty()

st.subheader("Track history")
track_ph = st.empty()

st.subheader("Detected events across time")
heat_ph = st.empty()

if enable_logging:
    st.subheader("Recent event log (SQLite)")
    log_ph = st.empty()
else:
    log_ph = None


# ==============================================================================
# JAMMER PRESET LIBRARY
# ==============================================================================
ALL_JAMMER_TYPES = ["CW", "Chirp", "WidebandNoise", "Burst", "FHSS"]

# Bandwidth used by each jammer class in the simulator
JAMMER_BW = {
    "CW": 1e5,
    "Chirp": 2e6,
    "WidebandNoise": 3e6,
    "Burst": 1e5,
    "FHSS": 2e6,
}


def build_jammer_list(frame_i, n_jammers, base_amp):
    """
    Build a list of jammers for the current frame.
    - Cycles through all 5 jammer types
    - Each jammer gets a randomized frequency + amplitude
    - CW jammers stay stationary (good for tracking demo)
    - Other types drift or hop over time
    """
    rng = np.random.default_rng(frame_i)   # reproducible per frame
    jammers = []

    # Fixed frequency slots spread across -8 to +8 MHz
    slots = np.linspace(-8e6, 8e6, max(n_jammers, 1))

    for i in range(n_jammers):
        jtype = ALL_JAMMER_TYPES[i % len(ALL_JAMMER_TYPES)]
        bw = JAMMER_BW[jtype] * rng.uniform(0.7, 1.3)
        amp = base_amp * rng.uniform(0.5, 1.0)

        # Frequency: some jammers are stationary, others drift
        if jtype == "CW":
            f = slots[i]                                # stationary
        elif jtype == "FHSS":
            f = slots[i] + 0.4e6 * np.sin(frame_i * 0.15 + i)
        elif jtype == "Chirp":
            f = slots[i] + 1.0e6 * np.sin(frame_i * 0.1 + i)
        else:
            f = slots[i] + 0.2e6 * np.sin(frame_i * 0.05 + i)

        # Clip to stay within -10 to +10 MHz band
        f = float(np.clip(f, -9.5e6, 9.5e6))

        jammers.append({
            "type": jtype,
            "f": f,
            "bw": bw,
            "amp": amp,
        })

    return jammers


# ==============================================================================
# MAIN LOOP
# ==============================================================================
frame_i = 0
while True:
    frame_i += 1

    # -------------------------------------------------------------------------
    # 1. GET IQ SAMPLES (Phase 11: real SDR or simulation)
    # -------------------------------------------------------------------------
    if sdr is not None and sdr.available:
        iq = sdr.read_iq(N_FFT)
    else:
        jammers = build_jammer_list(frame_i, n_jammers, jam_amp)
        iq = simulate_frame(N_FFT, FS, jammers=jammers, seed=1000 + frame_i)

    # -------------------------------------------------------------------------
    # 2. COMPUTE PSD (Phase 9 upgrade: BH window + median filter)
    # -------------------------------------------------------------------------
    if use_bh:
        freqs, p_lin, p_db = compute_linear_psd_v2(iq, FS, use_bh=True)
    else:
        freqs, p_lin, p_db = compute_linear_psd(iq, FS)

    # -------------------------------------------------------------------------
    # 3. CFAR DETECTION (Phase 8: mode selectable)
    # -------------------------------------------------------------------------
    if cfar_mode == "CA-CFAR":
        thr_lin = linear_ca_cfar(p_lin, offset_db=cfar_offset)
    elif cfar_mode == "OS-CFAR":
        thr_lin = os_cfar(p_lin, offset_db=cfar_offset)
    else:  # VI-CFAR
        thr_lin = vi_cfar(p_lin, offset_db=cfar_offset)

    thr_db = 10 * np.log10(thr_lin + 1e-20) - 10 * np.log10(np.max(p_lin))

    # -------------------------------------------------------------------------
    # 4. DETECTION MASK (Phase 9 morphology)
    # -------------------------------------------------------------------------
    raw_mask = p_db > thr_db
    if use_morph:
        raw_mask = clean_detection_mask(raw_mask, close_size=5, open_size=3)
    det = np.where(raw_mask)[0]

    # -------------------------------------------------------------------------
    # 5. EVENT EXTRACTION (Phase 2)
    # -------------------------------------------------------------------------
    events, inter_bins = extract_events(det, freqs, p_db, thr_db, cfar_offset)

    # -------------------------------------------------------------------------
    # 6. CLASSIFICATION (Phase 5 RF / Phase 10 XGB / Phase 13 CNN)
    # -------------------------------------------------------------------------
    if classifier_choice == "1D-CNN (deep learning)" and cnn_model is not None:
        frame_label, frame_conf = classify_psd_cnn(p_db, cnn_model, cnn_le)
        for e in events:
            e["label"] = frame_label
            e["confidence"] = frame_conf
    elif classifier_choice == "XGBoost" and xgb_clf is not None:
        events = classify_events_xgb(events, p_db, freqs, thr_db,
                                     xgb_clf, xgb_sc, xgb_le, xgb_iso)
    else:
        events = classify_events(events, p_db, freqs, thr_db, rf_clf, rf_sc)

    # -------------------------------------------------------------------------
    # 7. FRAME BUFFER
    # -------------------------------------------------------------------------
    st.session_state.frames.append({
        "p_db": p_db.copy(),
        "thr_db": thr_db.copy(),
        "events": events,
        "det_bins": det,
    })
    if len(st.session_state.frames) > n_display:
        st.session_state.frames.pop(0)

    # -------------------------------------------------------------------------
    # 8. KALMAN TRACKING (Phase 6)
    # -------------------------------------------------------------------------
    if enable_track:
        dets_per_frame = [f["events"] for f in st.session_state.frames]
        st.session_state.tracks = track_frames(dets_per_frame, dt=update_rate)

    # -------------------------------------------------------------------------
    # 9. Pd / Pfa EVALUATION (Phase 3)
    # -------------------------------------------------------------------------
    Pd, Pfa, _, _ = evaluate_pd_pfa(events, det, freqs, GROUND_TRUTH_JAMMERS_MHZ)
    st.session_state.pd_list.append(Pd)
    st.session_state.pfa_list.append(Pfa)
    if len(st.session_state.pd_list) > 300:
        st.session_state.pd_list.pop(0)
        st.session_state.pfa_list.pop(0)

    # -------------------------------------------------------------------------
    # 10. SQLITE LOGGING (Phase 12)
    # -------------------------------------------------------------------------
    if enable_logging and events:
        try:
            log_events_to_db(st.session_state.db_conn, events)
        except Exception as e:
            st.warning(f"DB log error: {e}")

    # -------------------------------------------------------------------------
    # 11. CCI ALERT CHECK (Phase 12)
    # -------------------------------------------------------------------------
    intruders = check_protected_band_intrusion(events)
    if intruders:
        st.session_state.alert_state = True
        freqs_str = ', '.join(f"{e['f_center']:.3f} MHz" for e in intruders)
        alert_ph.error(
            f"🚨 **CCI ALERT**: {len(intruders)} interferer(s) inside the "
            f"protected band! Frequencies: {freqs_str}"
        )
    else:
        st.session_state.alert_state = False
        alert_ph.success("✅ Protected band is clear — no co-channel interference")

    # -------------------------------------------------------------------------
    # 12. WATERFALL PLOT
    # -------------------------------------------------------------------------
    wf = np.array([f["p_db"] for f in st.session_state.frames])
    fig_wf = go.Figure(go.Heatmap(
        z=wf, x=freqs / 1e6, y=np.arange(len(wf)),
        colorscale="Viridis", zmin=-90, zmax=0,
        colorbar=dict(title="dB")))
    fig_wf.update_layout(
        title=f"Waterfall ({len(wf)} frames, {n_jammers} jammers)",
        xaxis_title="Frequency (MHz)", yaxis_title="Frame",
        height=380, margin=dict(l=40, r=10, t=40, b=40))
    place_wf.plotly_chart(fig_wf, use_container_width=True,
                          key=f"wf_{frame_i}")

    # -------------------------------------------------------------------------
    # 13. PSD + CFAR PLOT
    # -------------------------------------------------------------------------
    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(
        x=freqs / 1e6, y=p_db, name="PSD",
        line=dict(color="#1f77b4", width=1)))
    fig_psd.add_trace(go.Scatter(
        x=freqs / 1e6, y=thr_db, name=f"{cfar_mode} Threshold",
        line=dict(color="#ff7f0e", dash="dash", width=1.2)))
    if len(inter_bins):
        fig_psd.add_trace(go.Scatter(
            x=freqs[inter_bins] / 1e6, y=p_db[inter_bins],
            mode="markers", name="Detected",
            marker=dict(color="red", size=5)))

    fmin = (PROTECTED_CENTER - PROTECTED_BANDWIDTH / 2) / 1e6
    fmax = (PROTECTED_CENTER + PROTECTED_BANDWIDTH / 2) / 1e6
    fig_psd.add_vrect(x0=fmin, x1=fmax, fillcolor="#17becf",
                      opacity=0.15, line_width=0,
                      annotation_text="Desired",
                      annotation_position="top left")

    fig_psd.update_layout(
        title=f"Power Spectrum + {cfar_mode}",
        xaxis_title="Frequency (MHz)",
        yaxis_title="Normalized Power (dB)",
        yaxis_range=[-100, 5], height=380,
        margin=dict(l=40, r=10, t=40, b=40), showlegend=True)
    place_psd.plotly_chart(fig_psd, use_container_width=True,
                           key=f"psd_{frame_i}")

    # -------------------------------------------------------------------------
    # 14. KPIs
    # -------------------------------------------------------------------------
    kpi_events.metric("Events detected", len(events))
    kpi_pd.metric("Pd (rolling)",
                  f"{np.mean(st.session_state.pd_list):.2f}")
    kpi_pfa.metric("Pfa (rolling)",
                   f"{np.mean(st.session_state.pfa_list):.4f}")
    active_tracks = len([t for t in st.session_state.tracks if t.misses == 0])
    kpi_trk.metric("Active tracks", active_tracks)

    # -------------------------------------------------------------------------
    # 15. EVENT TABLE
    # -------------------------------------------------------------------------
    if events:
        df = pd.DataFrame([{
            "Label": e["label"],
            "Conf": f"{e['confidence'] * 100:.1f}%",
            "Fc (MHz)": f"{e['f_center']:.3f}",
            "BW (MHz)": f"{e['bw']:.4f}",
            "Peak (dB)": f"{e['peak']:.2f}",
            "SNR (dB)": f"{e['snr']:.2f}",
        } for e in events])
        tbl_ph.dataframe(df, use_container_width=True)
    else:
        tbl_ph.info("No interference detected in current frame.")

    # -------------------------------------------------------------------------
    # 16. TRACK PLOT
    # -------------------------------------------------------------------------
    if enable_track and st.session_state.tracks:
        fig_tr = go.Figure()
        for tr in st.session_state.tracks:
            # Lower threshold so tracks appear almost immediately
            if tr.hits < 2:
                continue
            hist = tr.history
            fig_tr.add_trace(go.Scatter(
                x=list(range(len(hist))),
                y=[h[0] for h in hist],
                mode="lines+markers",
                name=f"#{tr.id} {tr.label}"))
        fig_tr.update_layout(
            xaxis_title="Frame",
            yaxis_title="Center freq (MHz)",
            height=320,
            margin=dict(l=40, r=10, t=10, b=40))
        track_ph.plotly_chart(fig_tr, use_container_width=True,
                              key=f"trk_{frame_i}")

    # -------------------------------------------------------------------------
    # 17. EVENT HEATMAP
    # -------------------------------------------------------------------------
    if len(st.session_state.frames) > 1:
        ev_mat = np.zeros((len(st.session_state.frames), len(freqs)))
        for i, fr in enumerate(st.session_state.frames):
            for e in fr["events"]:
                ev_mat[i, e["cluster"]] = e["snr"]
        fig_h = go.Figure(go.Heatmap(
            z=ev_mat, x=freqs / 1e6, y=np.arange(ev_mat.shape[0]),
            colorscale="Hot", colorbar=dict(title="SNR (dB)")))
        fig_h.update_layout(
            xaxis_title="Frequency (MHz)", yaxis_title="Frame",
            height=260, margin=dict(l=40, r=10, t=10, b=40))
        heat_ph.plotly_chart(fig_h, use_container_width=True,
                             key=f"heat_{frame_i}")

    # -------------------------------------------------------------------------
    # 18. SQLITE LOG TABLE (Phase 12)
    # -------------------------------------------------------------------------
    if enable_logging and log_ph is not None:
        try:
            rows = query_recent_events(st.session_state.db_conn, limit=20)
            if rows:
                df_log = pd.DataFrame(
                    rows,
                    columns=["Time", "Fc (MHz)", "BW (MHz)",
                             "Peak (dB)", "SNR (dB)", "Label", "Conf"])
                log_ph.dataframe(df_log, use_container_width=True)
            else:
                log_ph.info("No events logged yet.")
        except Exception as e:
            log_ph.warning(f"DB read error: {e}")

    # -------------------------------------------------------------------------
    # 19. SLEEP
    # -------------------------------------------------------------------------
    time.sleep(update_rate)