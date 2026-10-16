"""
RF Interference Detection Pipeline - Extended (Phase 1 to Phase 12)
Automatic Detection of Communication Signal Interference using
Adaptive CA-CFAR Thresholding, Event Extraction, and RF Feature Measurement.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import pickle
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal.windows import hann, blackmanharris
from scipy.signal import stft as scipy_stft
from scipy.stats import kurtosis, skew
from scipy.ndimage import median_filter, binary_closing, binary_opening

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)

# Optional imports (graceful fallback if not installed)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[!] XGBoost not installed. Phase 10 (XGBoost) will be unavailable.")

try:
    from rtlsdr import RtlSdr
    SDR_AVAILABLE = True
except ImportError:
    SDR_AVAILABLE = False


# ==============================================================================
# CONFIG
# ==============================================================================
FS = 20.0e6
N_FFT = 8192
NUM_GUARD = 4
NUM_REF = 20
CFAR_OFFSET_DB = 15.0
PROTECTED_CENTER = 2.0e6
PROTECTED_BANDWIDTH = 0.30e6

# Ground-truth jammers used by the simulator (for Pd/Pfa)
GROUND_TRUTH_JAMMERS_MHZ = [-3.5, 6.0]

# Interference classes for the classifier
CLASSES = ["CW", "Chirp", "WidebandNoise", "Burst", "FHSS"]

# Model paths (Phases 5, 10)
MODEL_PATH = "rf_classifier.pkl"
SCALER_PATH = "rf_scaler.pkl"

# Extended config (Phases 10, 12)
XGB_MODEL_PATH = "rf_xgb_model.pkl"
XGB_SCALER_PATH = "rf_xgb_scaler.pkl"
XGB_ENCODER_PATH = "rf_xgb_encoder.pkl"
UNKNOWN_MODEL_PATH = "rf_unknown_detector.pkl"
DB_PATH = "rf_events.db"


# ==============================================================================
# PHASE 1-2: DATA LOADING, SIMULATION, PSD, CA-CFAR, EVENT EXTRACTION
# ==============================================================================
def load_or_synthesize_iq(file_path=None, n_samples=N_FFT, fs=FS):
    if file_path and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            i_col = next((c for c in cols if c in
                          ['i', 'inphase', 'real', 'i_data', 're', 'ch1']), None)
            q_col = next((c for c in cols if c in
                          ['q', 'quadrature', 'imag', 'imaginary',
                           'q_data', 'im', 'ch2']), None)
            if i_col and q_col:
                i_vals = pd.to_numeric(df[i_col], errors='coerce').fillna(0).values[:n_samples]
                q_vals = pd.to_numeric(df[q_col], errors='coerce').fillna(0).values[:n_samples]
                iq = i_vals + 1j * q_vals
            elif df.shape[1] >= 2 and np.issubdtype(df.iloc[:, 0].dtype, np.number):
                i_vals = pd.to_numeric(df.iloc[:n_samples, 0], errors='coerce').fillna(0).values
                q_vals = pd.to_numeric(df.iloc[:n_samples, 1], errors='coerce').fillna(0).values
                iq = i_vals + 1j * q_vals
            else:
                col = df.iloc[:n_samples, 0]
                iq = (col.astype(str).str.replace('i', 'j').apply(complex).values
                      if col.dtype == object else col.values.astype(np.complex64))
            if len(iq) < n_samples:
                iq = np.pad(iq, (0, n_samples - len(iq)))
            return iq[:n_samples]
        except Exception as e:
            print(f"[!] CSV parse failed: {e}. Using simulation.")
    return simulate_frame(n_samples, fs)


def simulate_frame(n_samples=N_FFT, fs=FS, jammers=None,
                   noise_sigma=0.004, seed=None):
    """Simulate one IQ frame. `jammers` is list of dicts:
       {'type': 'CW'|'Chirp'|'WidebandNoise'|'Burst'|'FHSS',
        'f': center_Hz, 'bw': bandwidth_Hz, 'amp': amplitude}"""
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(n_samples) / fs

    # Desired signal
    sig = 0.25 * np.exp(1j * 2 * np.pi * PROTECTED_CENTER * t)

    # Default jammers if none supplied
    if jammers is None:
        jammers = [
            {"type": "CW", "f": -3.5e6, "bw": 1e5, "amp": 1.00},
            {"type": "CW", "f": 6.0e6, "bw": 1e5, "amp": 0.45},
        ]

    for j in jammers:
        f, bw, amp, kind = j["f"], j["bw"], j["amp"], j["type"]
        if kind == "CW":
            sig += amp * np.exp(1j * 2 * np.pi * f * t)
        elif kind == "Chirp":
            k = bw / (n_samples / fs)
            phase = 2 * np.pi * (f * t + 0.5 * k * t ** 2)
            sig += amp * np.exp(1j * phase)
        elif kind == "WidebandNoise":
            n = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
            Nf = np.fft.fft(n)
            f_ax = np.fft.fftfreq(n_samples, 1 / fs)
            mask = (np.abs(f_ax - f) <= bw / 2)
            Nf = Nf * mask
            sig += amp * np.fft.ifft(Nf)
        elif kind == "Burst":
            env = (np.sin(2 * np.pi * 2000 * t) > 0).astype(float)
            sig += amp * env * np.exp(1j * 2 * np.pi * f * t)
        elif kind == "FHSS":
            hop_bins = np.linspace(f - bw, f + bw, 5)
            hop_idx = (np.arange(n_samples) // (n_samples // 5)) % 5
            phase = 2 * np.pi * hop_bins[hop_idx] * t
            sig += amp * np.exp(1j * phase)

    noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * \
            (noise_sigma / np.sqrt(2))
    return sig + noise


def compute_linear_psd(iq_signal, fs=FS):
    n = len(iq_signal)
    win = hann(n, sym=False)
    iq_win = iq_signal * win
    spectrum = np.fft.fftshift(np.fft.fft(iq_win))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs))
    power_lin = np.abs(spectrum) ** 2 / np.sum(win ** 2)
    power_db = 10 * np.log10(power_lin + 1e-20)
    power_db_norm = power_db - np.max(power_db)
    return freqs, power_lin, power_db_norm


def linear_ca_cfar(power_lin, num_guard=NUM_GUARD, num_ref=NUM_REF,
                   offset_db=CFAR_OFFSET_DB):
    scale = 10.0 ** (offset_db / 10.0)
    n = len(power_lin)
    thr = np.zeros(n)
    for cut in range(n):
        ls, le = cut - num_guard - num_ref, cut - num_guard
        rs, re = cut + num_guard + 1, cut + num_guard + 1 + num_ref
        ref = []
        if ls >= 0:
            ref.extend(power_lin[ls:le])
        elif le > 0:
            ref.extend(power_lin[0:le])
        if re <= n:
            ref.extend(power_lin[rs:re])
        elif rs < n:
            ref.extend(power_lin[rs:n])
        noise = np.mean(ref) if ref else np.mean(power_lin)
        thr[cut] = noise * scale
    return thr


def extract_events(raw_bins, freqs, power_db_norm, threshold_db_norm, offset_db):
    f_min = PROTECTED_CENTER - PROTECTED_BANDWIDTH / 2
    f_max = PROTECTED_CENTER + PROTECTED_BANDWIDTH / 2
    inter_bins = [b for b in raw_bins if not (f_min <= freqs[b] <= f_max)]

    clusters = []
    if inter_bins:
        cur = [inter_bins[0]]
        for b in inter_bins[1:]:
            if b == cur[-1] + 1:
                cur.append(b)
            else:
                clusters.append(cur)
                cur = [b]
        clusters.append(cur)

    bin_width = freqs[1] - freqs[0]
    out = []
    for i, cl in enumerate(clusters, 1):
        cl = np.array(cl)
        peak_idx = cl[np.argmax(power_db_norm[cl])]
        peak = power_db_norm[peak_idx]
        local_noise = threshold_db_norm[peak_idx] - offset_db
        out.append({
            "event": i,
            "cluster": cl,
            "f_center": freqs[peak_idx] / 1e6,
            "bw": len(cl) * bin_width / 1e6,
            "peak": peak,
            "snr": peak - local_noise,
        })
    return out, np.array(inter_bins)


# ==============================================================================
# PHASE 3: Pd / Pfa EVALUATION + ROC
# ==============================================================================
def evaluate_pd_pfa(event_list, detected_bins, freqs,
                    ground_truth_mhz, tol_mhz=0.15, guard_bins=40):
    detected_freqs = np.array([e["f_center"] for e in event_list])

    tp = 0
    for gt in ground_truth_mhz:
        if len(detected_freqs) and np.min(np.abs(detected_freqs - gt)) < tol_mhz:
            tp += 1
    fn = len(ground_truth_mhz) - tp
    Pd = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    gt_bins = set()
    for gt in ground_truth_mhz:
        idx = int(np.argmin(np.abs(freqs / 1e6 - gt)))
        gt_bins.update(range(max(0, idx - guard_bins),
                             min(len(freqs), idx + guard_bins)))

    f_min = (PROTECTED_CENTER - PROTECTED_BANDWIDTH / 2) / 1e6
    f_max = (PROTECTED_CENTER + PROTECTED_BANDWIDTH / 2) / 1e6
    prot_bins = set(np.where((freqs / 1e6 >= f_min) & (freqs / 1e6 <= f_max))[0])

    false_bins = [b for b in detected_bins if b not in gt_bins and b not in prot_bins]
    valid_bins = len(freqs) - len(gt_bins) - len(prot_bins)
    Pfa = len(false_bins) / valid_bins if valid_bins > 0 else 0.0

    return Pd, Pfa, tp, fn


def snr_sweep_pd_pfa(snr_db_list=np.arange(0, 26, 2), n_trials=50,
                     jammer_mhz=None, save_plot=True):
    if jammer_mhz is None:
        jammer_mhz = GROUND_TRUTH_JAMMERS_MHZ

    Pd_curve, Pfa_curve = [], []

    for snr_db in snr_db_list:
        amp = 10 ** (snr_db / 20.0) * 0.004
        pd_acc, pfa_acc = [], []
        for trial in range(n_trials):
            jammers = [{"type": "CW", "f": f * 1e6, "bw": 1e5, "amp": amp}
                       for f in jammer_mhz]
            iq = simulate_frame(N_FFT, FS, jammers=jammers,
                                noise_sigma=0.004, seed=1000 + trial)
            freqs, p_lin, p_db = compute_linear_psd(iq, FS)
            thr_lin = linear_ca_cfar(p_lin)
            thr_db = 10 * np.log10(thr_lin + 1e-20) - 10 * np.log10(np.max(p_lin))
            det_bins = np.where(p_db > thr_db)[0]
            events, inter_bins = extract_events(det_bins, freqs, p_db, thr_db,
                                                CFAR_OFFSET_DB)
            Pd, Pfa, _, _ = evaluate_pd_pfa(events, det_bins, freqs, jammer_mhz)
            pd_acc.append(Pd)
            pfa_acc.append(Pfa)
        Pd_curve.append(np.mean(pd_acc))
        Pfa_curve.append(np.mean(pfa_acc))
        print(f"  SNR={snr_db:2d} dB  ->  Pd={Pd_curve[-1]:.3f}  Pfa={Pfa_curve[-1]:.5f}")

    if save_plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
        ax[0].plot(snr_db_list, Pd_curve, 'o-', color='#2ca02c', label='Pd')
        ax[0].plot(snr_db_list, Pfa_curve, 's--', color='#d62728', label='Pfa')
        ax[0].set_xlabel("SNR (dB)")
        ax[0].set_ylabel("Probability")
        ax[0].set_title("Detection & False-Alarm vs SNR")
        ax[0].grid(True, ls='--', alpha=0.5)
        ax[0].legend()

        ax[1].plot(Pfa_curve, Pd_curve, 'o-', color='#1f77b4')
        ax[1].set_xlabel("Pfa")
        ax[1].set_ylabel("Pd")
        ax[1].set_title("ROC Curve")
        ax[1].grid(True, ls='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("phase3_roc.png", dpi=120)
        plt.show()

    return snr_db_list, Pd_curve, Pfa_curve


# ==============================================================================
# PHASE 4: 12-D FEATURE EXTRACTION
# ==============================================================================
def extract_features_12d(power_db_norm, freqs, cluster,
                         threshold_db_norm, noise_floor_db):
    cl = np.asarray(cluster)
    p = power_db_norm[cl]
    p_lin = 10 ** (p / 10.0)

    peak_idx = cl[np.argmax(p)]
    peak = power_db_norm[peak_idx]
    mean_p = float(np.mean(p))
    snr = peak - noise_floor_db

    bin_width_mhz = abs(freqs[1] - freqs[0]) / 1e6
    bw_mhz = len(cl) * bin_width_mhz
    f_center_mhz = float(freqs[peak_idx] / 1e6)

    tot = np.sum(p_lin) + 1e-20
    p_norm = p_lin / tot
    f_local = np.linspace(-0.5, 0.5, len(cl))
    f_mean = np.sum(f_local * p_norm)
    f_var = np.sum((f_local - f_mean) ** 2 * p_norm)
    f_std = np.sqrt(f_var + 1e-20)
    sk = np.sum(((f_local - f_mean) / f_std) ** 3 * p_norm)
    ku = np.sum(((f_local - f_mean) / f_std) ** 4 * p_norm)

    p_pos = np.clip(p_lin, 1e-20, None)
    flatness = np.exp(np.mean(np.log(p_pos))) / (np.mean(p_pos) + 1e-20)
    pmr = peak - mean_p

    cum = np.cumsum(p_lin) / (np.sum(p_lin) + 1e-20)
    idx_lo = int(np.searchsorted(cum, 0.005))
    idx_hi = int(np.searchsorted(cum, 0.995))
    obw_mhz = (idx_hi - idx_lo) * bin_width_mhz

    if len(p) >= 3:
        edge = (np.abs(p[1] - p[0]) + np.abs(p[-1] - p[-2])) / 2.0
    else:
        edge = 0.0

    full = power_db_norm
    left = full[max(0, cl[0] - 2):cl[0]]
    right = full[cl[-1] + 1:cl[-1] + 3]
    adj = np.concatenate([left, right]) if (len(left) + len(right)) else np.array([mean_p])
    aclr = mean_p - float(np.mean(adj))

    return np.array([
        f_center_mhz, bw_mhz, peak, mean_p, snr, flatness,
        ku, sk, pmr, edge, obw_mhz, aclr,
    ], dtype=np.float32)


FEATURE_NAMES = ["f_center", "bw", "peak", "mean_p", "snr", "flatness",
                 "kurtosis", "skewness", "pmr", "edge", "obw", "aclr"]


def build_event_dataset(n_per_class=150, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []

    for cls in CLASSES:
        print(f"  Generating {n_per_class} '{cls}' events ...")
        for k in range(n_per_class):
            f0 = rng.uniform(-8e6, 8e6)
            amp = rng.uniform(0.3, 1.2)
            if cls == "CW":
                bw = rng.uniform(3e4, 2e5)
                j = {"type": "CW", "f": f0, "bw": bw, "amp": amp}
            elif cls == "Chirp":
                bw = rng.uniform(1e6, 3e6)
                j = {"type": "Chirp", "f": f0, "bw": bw, "amp": amp}
            elif cls == "WidebandNoise":
                bw = rng.uniform(2e6, 5e6)
                j = {"type": "WidebandNoise", "f": f0, "bw": bw, "amp": amp}
            elif cls == "Burst":
                bw = rng.uniform(5e4, 2e5)
                j = {"type": "Burst", "f": f0, "bw": bw, "amp": amp}
            else:  # FHSS
                bw = rng.uniform(1e6, 3e6)
                j = {"type": "FHSS", "f": f0, "bw": bw, "amp": amp}

            iq = simulate_frame(N_FFT, FS, jammers=[j], noise_sigma=0.004,
                                seed=int(rng.integers(0, 1e6)))
            freqs, p_lin, p_db = compute_linear_psd(iq, FS)
            thr_lin = linear_ca_cfar(p_lin)
            thr_db = 10 * np.log10(thr_lin + 1e-20) - 10 * np.log10(np.max(p_lin))
            det = np.where(p_db > thr_db)[0]
            events, _ = extract_events(det, freqs, p_db, thr_db, CFAR_OFFSET_DB)

            for e in events:
                noise_floor = float(thr_db[e["cluster"][0]]) - CFAR_OFFSET_DB
                feat = extract_features_12d(p_db, freqs, e["cluster"],
                                            thr_db, noise_floor)
                X.append(feat)
                y.append(cls)

    return np.array(X, dtype=np.float32), np.array(y)


# ==============================================================================
# PHASE 5: RANDOM FOREST CLASSIFIER
# ==============================================================================
def train_classifier(X, y, save=True):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2,
                                          random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                 random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)

    y_pred = clf.predict(Xte)
    print("\n=== Random Forest Classification Report (holdout) ===")
    print(classification_report(yte, y_pred, zero_division=0))

    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    print("\n=== Feature importance ===")
    for i in order:
        print(f"  {FEATURE_NAMES[i]:<10s}  {imp[i]:.4f}")

    if save:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(clf, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"\nSaved: {MODEL_PATH}, {SCALER_PATH}")

    return clf, scaler


def load_classifier():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            clf = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            sc = pickle.load(f)
        return clf, sc
    return None, None


def classify_events(event_list, power_db_norm, freqs,
                    threshold_db_norm, clf, scaler):
    for e in event_list:
        noise_floor = float(threshold_db_norm[e["cluster"][0]]) - CFAR_OFFSET_DB
        feat = extract_features_12d(power_db_norm, freqs, e["cluster"],
                                    threshold_db_norm, noise_floor)
        fs_ = scaler.transform(feat.reshape(1, -1))
        proba = clf.predict_proba(fs_)[0]
        idx = int(np.argmax(proba))
        e["label"] = clf.classes_[idx]
        e["confidence"] = float(proba[idx])
    return event_list


# ==============================================================================
# PHASE 6: KALMAN TRACKING
# ==============================================================================
class KalmanTrack:
    _next_id = 1

    def __init__(self, f0, p0, dt):
        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        self.x = np.array([f0, 0.0])
        self.P = np.eye(2) * 1.0
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.diag([1e-3, 1e-2])
        self.R = np.array([[0.005]])
        self.power = p0
        self.age = 0
        self.misses = 0
        self.hits = 1
        self.history = [(f0, p0)]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1

    def update(self, z, p):
        y = z - (self.H @ self.x)[0]
        S = (self.H @ self.P @ self.H.T + self.R)[0, 0]
        K = (self.P @ self.H.T) / S
        self.x = self.x + (K.flatten() * y)
        self.P = (np.eye(2) - K @ self.H) @ self.P
        self.misses = 0
        self.hits += 1
        self.power = p
        self.history.append((float(self.x[0]), p))


def track_frames(detections_per_frame, dt=0.05, gate_mhz=0.5, max_misses=5):
    tracks = []
    for frame_dets in detections_per_frame:
        for tr in tracks:
            tr.predict()

        used_det = set()
        for tr in tracks:
            best_d, best_i = None, -1
            for i, d in enumerate(frame_dets):
                if i in used_det:
                    continue
                dist = abs(d["f_center"] - tr.x[0])
                if dist < gate_mhz and (best_d is None or dist < best_d):
                    best_d, best_i = dist, i
            if best_i >= 0:
                d = frame_dets[best_i]
                tr.update(d["f_center"], d["peak"])
                tr.label = d.get("label", getattr(tr, "label", "?"))
                used_det.add(best_i)
            else:
                tr.misses += 1

        for i, d in enumerate(frame_dets):
            if i in used_det:
                continue
            t = KalmanTrack(d["f_center"], d["peak"], dt)
            t.label = d.get("label", "?")
            tracks.append(t)

        tracks = [t for t in tracks if t.misses <= max_misses]

    return tracks


# ==============================================================================
# PHASE 8: ADVANCED CFAR VARIANTS (OS-CFAR, VI-CFAR, 2D-CFAR)
# ==============================================================================
def os_cfar(power_lin, num_guard=NUM_GUARD, num_ref=NUM_REF,
            offset_db=CFAR_OFFSET_DB, k_rank=None):
    scale = 10.0 ** (offset_db / 10.0)
    n = len(power_lin)
    n_ref_total = 2 * num_ref
    if k_rank is None:
        k_rank = int(0.75 * n_ref_total)

    thr = np.zeros(n)
    for cut in range(n):
        ls, le = cut - num_guard - num_ref, cut - num_guard
        rs, re = cut + num_guard + 1, cut + num_guard + 1 + num_ref
        ref = []
        if ls >= 0:
            ref.extend(power_lin[ls:le])
        elif le > 0:
            ref.extend(power_lin[0:le])
        if re <= n:
            ref.extend(power_lin[rs:re])
        elif rs < n:
            ref.extend(power_lin[rs:n])
        if len(ref) < 3:
            thr[cut] = np.mean(power_lin) * scale
            continue
        ref_sorted = np.sort(ref)
        idx = min(k_rank, len(ref_sorted) - 1)
        thr[cut] = ref_sorted[idx] * scale
    return thr


def vi_cfar(power_lin, num_guard=NUM_GUARD, num_ref=NUM_REF,
            offset_db=CFAR_OFFSET_DB):
    scale = 10.0 ** (offset_db / 10.0)
    n = len(power_lin)
    thr = np.zeros(n)

    for cut in range(n):
        ls, le = max(0, cut - num_guard - num_ref), max(0, cut - num_guard)
        rs, re = min(n, cut + num_guard + 1), min(n, cut + num_guard + 1 + num_ref)
        left = power_lin[ls:le]
        right = power_lin[rs:re]

        if len(left) < 3 or len(right) < 3:
            thr[cut] = np.mean(power_lin) * scale
            continue

        mL, vL = np.mean(left), np.var(left)
        mR, vR = np.mean(right), np.var(right)

        VI_L = 1 + vL / (mL ** 2 + 1e-20)
        VI_R = 1 + vR / (mR ** 2 + 1e-20)
        MR_L = mL / (np.median(left) + 1e-20)
        MR_R = mR / (np.median(right) + 1e-20)

        VI_thresh, MR_thresh = 4.76, 1.8

        if VI_L < VI_thresh and VI_R < VI_thresh:
            if MR_L < MR_thresh and MR_R < MR_thresh:
                noise = (mL + mR) / 2
            else:
                noise = min(mL, mR)
        else:
            noise = max(mL, mR)

        thr[cut] = noise * scale
    return thr


def cfar_2d(power_db_2d, num_guard=2, num_ref=8, offset_db=CFAR_OFFSET_DB):
    scale = 10.0 ** (offset_db / 10.0)
    T, F = power_db_2d.shape
    power_lin = 10 ** (power_db_2d / 10.0)
    thr = np.zeros_like(power_lin)

    for t in range(T):
        for f in range(F):
            t0, t1 = max(0, t - num_ref - num_guard), max(0, t - num_guard)
            t2, t3 = min(T, t + num_guard + 1), min(T, t + num_guard + 1 + num_ref)
            f0, f1 = max(0, f - num_ref - num_guard), max(0, f - num_guard)
            f2, f3 = min(F, f + num_guard + 1), min(F, f + num_guard + 1 + num_ref)
            ref = np.concatenate([
                power_lin[t0:t1, f0:f3].ravel(),
                power_lin[t2:t3, f0:f3].ravel(),
                power_lin[t0:t3, f0:f1].ravel(),
                power_lin[t0:t3, f2:f3].ravel(),
            ])
            noise = np.mean(ref) if len(ref) else np.mean(power_lin)
            thr[t, f] = noise * scale
    return thr


# ==============================================================================
# PHASE 9: WINDOW UPGRADE + MORPHOLOGICAL CLEANUP
# ==============================================================================
def compute_linear_psd_v2(iq_signal, fs=FS, use_bh=True, median_ksize=5):
    n = len(iq_signal)
    win = blackmanharris(n, sym=False) if use_bh else hann(n, sym=False)

    iq_win = iq_signal * win
    spectrum = np.fft.fftshift(np.fft.fft(iq_win))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs))
    power_lin = np.abs(spectrum) ** 2 / np.sum(win ** 2)
    power_db = 10 * np.log10(power_lin + 1e-20)

    power_db_smooth = median_filter(power_db, size=median_ksize)
    power_db_norm = power_db_smooth - np.max(power_db_smooth)
    power_lin_norm = 10 ** (power_db_norm / 10.0)

    return freqs, power_lin_norm, power_db_norm


def clean_detection_mask(mask, close_size=5, open_size=3):
    m = mask.astype(bool)
    if close_size > 1:
        m = binary_closing(m, structure=np.ones(close_size))
    if open_size > 1:
        m = binary_opening(m, structure=np.ones(open_size))
    return m


# ==============================================================================
# PHASE 10: XGBOOST + UNKNOWN JAMMER DETECTION
# ==============================================================================
def train_xgboost(X, y, save=True):
    if not XGB_AVAILABLE:
        raise RuntimeError("XGBoost is not installed. Run: pip install xgboost")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y_enc, test_size=0.2,
                                          random_state=42, stratify=y_enc)

    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        objective='multi:softprob', num_class=len(le.classes_),
        n_jobs=-1, random_state=42, eval_metric='mlogloss'
    )
    clf.fit(Xtr, ytr)

    y_pred = le.inverse_transform(clf.predict(Xte))
    y_true = le.inverse_transform(yte)
    print("\n=== XGBoost Classification Report ===")
    print(classification_report(y_true, y_pred, zero_division=0))

    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(Xs)

    if save:
        with open(XGB_MODEL_PATH, 'wb') as f:
            pickle.dump(clf, f)
        with open(XGB_SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        with open(XGB_ENCODER_PATH, 'wb') as f:
            pickle.dump(le, f)
        with open(UNKNOWN_MODEL_PATH, 'wb') as f:
            pickle.dump(iso, f)
        print(f"Saved: {XGB_MODEL_PATH}, {XGB_SCALER_PATH}, "
              f"{XGB_ENCODER_PATH}, {UNKNOWN_MODEL_PATH}")

    return clf, scaler, le, iso


def load_xgboost():
    if not XGB_AVAILABLE:
        return None, None, None, None
    if not all(os.path.exists(p) for p in
               [XGB_MODEL_PATH, XGB_SCALER_PATH, XGB_ENCODER_PATH]):
        return None, None, None, None
    with open(XGB_MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(XGB_SCALER_PATH, 'rb') as f:
        sc = pickle.load(f)
    with open(XGB_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    iso = None
    if os.path.exists(UNKNOWN_MODEL_PATH):
        with open(UNKNOWN_MODEL_PATH, 'rb') as f:
            iso = pickle.load(f)
    return clf, sc, le, iso


def classify_events_xgb(event_list, power_db_norm, freqs,
                        threshold_db_norm, clf, scaler, le, iso=None,
                        unknown_threshold=-0.05):
    for e in event_list:
        noise_floor = float(threshold_db_norm[e["cluster"][0]]) - CFAR_OFFSET_DB
        feat = extract_features_12d(power_db_norm, freqs, e["cluster"],
                                    threshold_db_norm, noise_floor)
        feat_s = scaler.transform(feat.reshape(1, -1))

        if iso is not None:
            score = iso.score_samples(feat_s)[0]
            if score < unknown_threshold:
                e["label"] = "UNKNOWN"
                e["confidence"] = 0.0
                continue

        proba = clf.predict_proba(feat_s)[0]
        idx = int(np.argmax(proba))
        e["label"] = le.inverse_transform([idx])[0]
        e["confidence"] = float(proba[idx])
    return event_list


# ==============================================================================
# PHASE 11: REAL SDR INTEGRATION (RTL-SDR)
# ==============================================================================
class RTLSDRIQSource:
    """Wrapper for RTL-SDR hardware. Gracefully falls back to simulation."""

    def __init__(self, center_freq=100e6, sample_rate=2.4e6, gain='auto'):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.sdr = None
        self.available = False

        if SDR_AVAILABLE:
            try:
                self.sdr = RtlSdr()
                self.sdr.sample_rate = sample_rate
                self.sdr.center_freq = center_freq
                self.sdr.gain = gain
                self.available = True
                print(f"[OK] RTL-SDR connected at {center_freq/1e6:.2f} MHz")
            except Exception as e:
                print(f"[!] RTL-SDR connection failed: {e}")
                print("    Falling back to simulation mode.")

    def read_iq(self, n_samples=N_FFT):
        if not self.available:
            return simulate_frame(n_samples, FS)
        try:
            samples = self.sdr.read_samples(n_samples)
            return samples.astype(np.complex64)
        except Exception as e:
            print(f"[!] SDR read error: {e}")
            return simulate_frame(n_samples, FS)

    def close(self):
        if self.sdr:
            try:
                self.sdr.close()
            except Exception:
                pass


# ==============================================================================
# PHASE 12: ALERT SYSTEM + SQLITE LOGGING
# ==============================================================================
def init_event_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            f_center_mhz REAL,
            bandwidth_mhz REAL,
            peak_db REAL,
            snr_db REAL,
            label TEXT,
            confidence REAL,
            in_protected_band INTEGER
        )
    """)
    conn.commit()
    return conn


def log_events_to_db(conn, events):
    t = datetime.now().isoformat(timespec='seconds')
    fmin = (PROTECTED_CENTER - PROTECTED_BANDWIDTH / 2) / 1e6
    fmax = (PROTECTED_CENTER + PROTECTED_BANDWIDTH / 2) / 1e6
    for e in events:
        in_band = int(fmin <= e["f_center"] <= fmax)
        conn.execute(
            "INSERT INTO events VALUES (NULL,?,?,?,?,?,?,?,?)",
            (t, e["f_center"], e["bw"], e["peak"], e["snr"],
             e.get("label", "?"), e.get("confidence", 0.0), in_band)
        )
    conn.commit()


def query_recent_events(conn, limit=100):
    cur = conn.execute(
        "SELECT timestamp, f_center_mhz, bandwidth_mhz, peak_db, snr_db, "
        "label, confidence FROM events ORDER BY id DESC LIMIT ?", (limit,)
    )
    return cur.fetchall()


def check_protected_band_intrusion(events):
    fmin = (PROTECTED_CENTER - PROTECTED_BANDWIDTH / 2) / 1e6
    fmax = (PROTECTED_CENTER + PROTECTED_BANDWIDTH / 2) / 1e6
    return [e for e in events if fmin <= e["f_center"] <= fmax]


# ==============================================================================
# DEMO / SELF-TEST
# ==============================================================================
def _demo():
    print("=" * 70)
    print("PHASE 3 : Pd / Pfa sweep")
    print("=" * 70)
    snr_sweep_pd_pfa(np.arange(0, 26, 2), n_trials=30)

    print("\n" + "=" * 70)
    print("PHASE 4-5 : Feature dataset + Random Forest classifier")
    print("=" * 70)
    X, y = build_event_dataset(n_per_class=100, seed=7)
    print(f"  Dataset: {X.shape}, classes: {np.unique(y, return_counts=True)}")
    clf, scaler = train_classifier(X, y)

    print("\n" + "=" * 70)
    print("PHASE 6 : Multi-frame tracking demo")
    print("=" * 70)
    frames = []
    for k in range(20):
        iq = simulate_frame(
            N_FFT, FS,
            jammers=[{"type": "CW", "f": -3.5e6, "bw": 1e5, "amp": 1.0},
                     {"type": "FHSS", "f": 5.5e6 + 0.1e6 * np.sin(k * 0.3),
                      "bw": 2e6, "amp": 0.5}],
            seed=100 + k)
        freqs, p_lin, p_db = compute_linear_psd(iq, FS)
        thr_lin = linear_ca_cfar(p_lin)
        thr_db = 10 * np.log10(thr_lin + 1e-20) - 10 * np.log10(np.max(p_lin))
        det = np.where(p_db > thr_db)[0]
        ev, _ = extract_events(det, freqs, p_db, thr_db, CFAR_OFFSET_DB)
        ev = classify_events(ev, p_db, freqs, thr_db, clf, scaler)
        frames.append(ev)

    tracks = track_frames(frames, dt=0.05)
    print(f"  Total tracks formed: {len(tracks)}")
    for tr in tracks:
        print(f"   Track #{tr.id:<2d}  hits={tr.hits:<2d} "
              f"label={tr.label:<15s} last_f={tr.x[0]:+.3f} MHz")

    plt.figure(figsize=(11, 4.5), dpi=120)
    for tr in tracks:
        if tr.hits < 3:
            continue
        f_hist = [h[0] for h in tr.history]
        plt.plot(range(len(f_hist)), f_hist, 'o-',
                 label=f"Track {tr.id} ({tr.label})")
    plt.xlabel("Frame")
    plt.ylabel("Center freq (MHz)")
    plt.title("Phase 6: Interference track evolution")
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("phase6_tracks.png", dpi=120)
    plt.show()

    print("\n" + "=" * 70)
    print("PHASE 8-12 : Advanced features demo")
    print("=" * 70)
    print(f"  XGBoost available: {XGB_AVAILABLE}")
    print(f"  RTL-SDR available: {SDR_AVAILABLE}")
    print("  Advanced CFAR, morphology, SQLite logging are ready to use.")


if __name__ == "__main__":
    _demo()