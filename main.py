import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal.windows import hann

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
DATASET_PATH = "logged_data.csv"

FS = 20.0e6               # Sampling rate: 20 MHz (-10 MHz to +10 MHz)
N_FFT = 8192              # FFT block size

# CA-CFAR Configuration
NUM_GUARD = 4             # Guard cells on each side
NUM_REF = 20              # Reference cells on each side
CFAR_OFFSET_DB = 15.0     # Threshold scale offset in dB

# Desired Band Protection
PROTECTED_CENTER = 2.0e6       # +2.0 MHz
PROTECTED_BANDWIDTH = 0.30e6   # +/- 0.15 MHz protected span


# ==============================================================================
# 1. DATA LOADER / CSV INGESTION
# ==============================================================================
def load_or_synthesize_iq(file_path=None, n_samples=N_FFT, fs=FS):

    if file_path and os.path.exists(file_path):

        print(f"[*] Found file: {file_path}. Parsing CSV...")

        try:
            df = pd.read_csv(file_path)

            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols

            i_candidates = [
                'i', 'inphase', 'real',
                'i_data', 're', 'ch1'
            ]

            q_candidates = [
                'q', 'quadrature', 'imag',
                'imaginary', 'q_data', 'im', 'ch2'
            ]

            i_col = next(
                (c for c in cols if c in i_candidates),
                None
            )

            q_col = next(
                (c for c in cols if c in q_candidates),
                None
            )

            # --------------------------------------------------------------
            # Case 1: Separate I and Q columns
            # --------------------------------------------------------------
            if i_col and q_col:

                print(
                    f"[*] Detected distinct IQ columns: "
                    f"Real='{i_col}', Imag='{q_col}'"
                )

                i_vals = pd.to_numeric(
                    df[i_col],
                    errors='coerce'
                ).fillna(0).values[:n_samples]

                q_vals = pd.to_numeric(
                    df[q_col],
                    errors='coerce'
                ).fillna(0).values[:n_samples]

                iq = i_vals + 1j * q_vals

            # --------------------------------------------------------------
            # Case 2: First two numerical columns are I and Q
            # --------------------------------------------------------------
            elif (
                df.shape[1] >= 2
                and np.issubdtype(
                    df.iloc[:, 0].dtype,
                    np.number
                )
            ):

                print(
                    "[*] Mapping first two numerical columns "
                    "directly to (I, Q)"
                )

                i_vals = pd.to_numeric(
                    df.iloc[:n_samples, 0],
                    errors='coerce'
                ).fillna(0).values

                q_vals = pd.to_numeric(
                    df.iloc[:n_samples, 1],
                    errors='coerce'
                ).fillna(0).values

                iq = i_vals + 1j * q_vals

            # --------------------------------------------------------------
            # Case 3: Single complex-valued column
            # --------------------------------------------------------------
            else:

                first_col = df.iloc[:n_samples, 0]

                if first_col.dtype == object:

                    iq = (
                        first_col
                        .astype(str)
                        .str.replace('i', 'j')
                        .apply(lambda x: complex(x))
                        .values
                    )

                else:

                    iq = first_col.values.astype(np.complex64)

            # --------------------------------------------------------------
            # Zero padding if data is shorter than FFT size
            # --------------------------------------------------------------
            if len(iq) < n_samples:

                print(
                    f"[!] Warning: CSV length ({len(iq)}) "
                    f"< N_FFT ({n_samples}). Zero-padding signal."
                )

                iq = np.pad(
                    iq,
                    (0, n_samples - len(iq)),
                    mode='constant'
                )

            return iq[:n_samples]

        except Exception as err:

            print(
                f"[!] Failed to parse CSV: {err}. "
                f"Falling back to simulation mode."
            )

    # ==========================================================================
    # SIMULATION MODE
    # ==========================================================================

    print(
        "[*] No valid CSV file found. "
        "Running Phase 2 simulation benchmark..."
    )

    np.random.seed(42)

    t = np.arange(n_samples) / fs

    # Desired communication signal
    desired = (
        0.25
        * np.exp(
            1j * 2 * np.pi * 2.0e6 * t
        )
    )

    # Narrowband interference / jammer 1
    jammer_1 = (
        1.00
        * np.exp(
            1j * 2 * np.pi * -3.5e6 * t
        )
    )

    # Narrowband interference / jammer 2
    jammer_2 = (
        0.45
        * np.exp(
            1j * 2 * np.pi * 6.0e6 * t
        )
    )

    # Complex Gaussian noise
    noise = (
        np.random.randn(n_samples)
        + 1j * np.random.randn(n_samples)
    ) * (0.004 / np.sqrt(2))

    return desired + jammer_1 + jammer_2 + noise


# ==============================================================================
# 2. SPECTRAL ESTIMATION: HANN WINDOW + PSD
# ==============================================================================
def compute_linear_psd(iq_signal, fs=FS):

    n = len(iq_signal)

    # Hann window
    win = hann(n, sym=False)

    # Apply window
    iq_win = iq_signal * win

    # FFT
    spectrum = np.fft.fftshift(
        np.fft.fft(iq_win)
    )

    # Frequency axis
    freqs = np.fft.fftshift(
        np.fft.fftfreq(
            n,
            d=1 / fs
        )
    )

    # Linear power
    power_linear = (
        np.abs(spectrum) ** 2
        / np.sum(win ** 2)
    )

    # Convert to dB
    power_db = 10 * np.log10(
        power_linear + 1e-20
    )

    # Normalize
    power_db_norm = (
        power_db
        - np.max(power_db)
    )

    return freqs, power_linear, power_db_norm


# ==============================================================================
# 3. LINEAR-POWER CA-CFAR DETECTOR
# ==============================================================================
def linear_ca_cfar(
    power_lin,
    num_guard=NUM_GUARD,
    num_ref=NUM_REF,
    offset_db=CFAR_OFFSET_DB
):

    # Convert dB offset to linear scale
    scale_factor = (
        10.0 ** (offset_db / 10.0)
    )

    n_bins = len(power_lin)

    threshold_lin = np.zeros(n_bins)

    # Process every FFT bin
    for cut in range(n_bins):

        left_start = (
            cut
            - num_guard
            - num_ref
        )

        left_end = (
            cut
            - num_guard
        )

        right_start = (
            cut
            + num_guard
            + 1
        )

        right_end = (
            cut
            + num_guard
            + 1
            + num_ref
        )

        ref_cells = []

        # --------------------------------------------------------------
        # Left reference cells
        # --------------------------------------------------------------
        if left_start >= 0:

            ref_cells.extend(
                power_lin[
                    left_start:left_end
                ]
            )

        elif left_end > 0:

            ref_cells.extend(
                power_lin[
                    0:left_end
                ]
            )

        # --------------------------------------------------------------
        # Right reference cells
        # --------------------------------------------------------------
        if right_end <= n_bins:

            ref_cells.extend(
                power_lin[
                    right_start:right_end
                ]
            )

        elif right_start < n_bins:

            ref_cells.extend(
                power_lin[
                    right_start:n_bins
                ]
            )

        # --------------------------------------------------------------
        # Noise estimation
        # --------------------------------------------------------------
        if len(ref_cells) > 0:

            noise_avg = np.mean(ref_cells)

        else:

            noise_avg = np.mean(power_lin)

        # --------------------------------------------------------------
        # CFAR threshold
        # --------------------------------------------------------------
        threshold_lin[cut] = (
            noise_avg
            * scale_factor
        )

    return threshold_lin


# ==============================================================================
# 4. DESIRED BAND PROTECTION & EVENT EXTRACTION
# ==============================================================================
def extract_events(
    raw_bins,
    freqs,
    power_db_norm,
    threshold_db_norm,
    offset_db
):

    # Protected desired band
    f_min = (
        PROTECTED_CENTER
        - (PROTECTED_BANDWIDTH / 2)
    )

    f_max = (
        PROTECTED_CENTER
        + (PROTECTED_BANDWIDTH / 2)
    )

    # Remove bins belonging to desired band
    interference_bins = [
        b for b in raw_bins
        if not (
            f_min
            <= freqs[b]
            <= f_max
        )
    ]

    events = []

    # --------------------------------------------------------------------------
    # Cluster adjacent detected bins
    # --------------------------------------------------------------------------
    if interference_bins:

        current_cluster = [
            interference_bins[0]
        ]

        for b in interference_bins[1:]:

            if b == current_cluster[-1] + 1:

                current_cluster.append(b)

            else:

                events.append(
                    current_cluster
                )

                current_cluster = [b]

        events.append(
            current_cluster
        )

    # --------------------------------------------------------------------------
    # Extract event information
    # --------------------------------------------------------------------------

    bin_width = (
        freqs[1]
        - freqs[0]
    )

    extracted = []

    for idx, cluster in enumerate(
        events,
        1
    ):

        cluster = np.array(cluster)

        # Peak bin
        peak_idx = cluster[
            np.argmax(
                power_db_norm[cluster]
            )
        ]

        # Center frequency
        center_f = (
            freqs[peak_idx]
            / 1e6
        )

        # Detected bandwidth
        bw = (
            len(cluster)
            * bin_width
            / 1e6
        )

        # Peak power
        peak_power = (
            power_db_norm[peak_idx]
        )

        # Local noise estimate
        local_noise = (
            threshold_db_norm[peak_idx]
            - offset_db
        )

        # SNR estimate
        snr = (
            peak_power
            - local_noise
        )

        extracted.append({

            "event": idx,

            "f_center": center_f,

            "bw": bw,

            "peak": peak_power,

            "snr": snr

        })

    return extracted, interference_bins


# ==============================================================================
# 5. EXECUTION & VISUALIZATION
# ==============================================================================
def main():

    # --------------------------------------------------------------------------
    # Load IQ data
    # --------------------------------------------------------------------------
    iq_samples = load_or_synthesize_iq(
        DATASET_PATH,
        n_samples=N_FFT,
        fs=FS
    )

    # --------------------------------------------------------------------------
    # Compute PSD
    # --------------------------------------------------------------------------
    freqs, power_linear, power_db_norm = (
        compute_linear_psd(
            iq_samples,
            fs=FS
        )
    )

    # --------------------------------------------------------------------------
    # CA-CFAR
    # --------------------------------------------------------------------------
    threshold_lin = linear_ca_cfar(
        power_linear,
        NUM_GUARD,
        NUM_REF,
        CFAR_OFFSET_DB
    )

    # Convert threshold to normalized dB
    threshold_db_norm = (
        10
        * np.log10(
            threshold_lin + 1e-20
        )
        - (
            10
            * np.log10(
                np.max(power_linear)
            )
        )
    )

    # --------------------------------------------------------------------------
    # Detect bins
    # --------------------------------------------------------------------------
    raw_detected_bins = np.where(
        power_db_norm
        > threshold_db_norm
    )[0]

    # --------------------------------------------------------------------------
    # Extract interference events
    # --------------------------------------------------------------------------
    event_list, interference_bins = (
        extract_events(
            raw_detected_bins,
            freqs,
            power_db_norm,
            threshold_db_norm,
            CFAR_OFFSET_DB
        )
    )

    # ==========================================================================
    # FINAL INTERFERENCE EVENT REPORT
    # ==========================================================================

    print("\n" + "=" * 72)

    if not event_list:

        print("NO INTERFERENCE DETECTED")

    else:

        for e in event_list:

            print(
                f"INTERFERENCE EVENT #{e['event']}"
            )

            print(
                f"Center frequency : "
                f"{e['f_center']:.3f} MHz"
            )

            print(
                f"Bandwidth        : "
                f"{e['bw']:.4f} MHz"
            )

            print(
                f"Peak power       : "
                f"{e['peak']:.2f} dB"
            )

            print(
                f"SNR              : "
                f"{e['snr']:.2f} dB"
            )

            print("=" * 72)

    print()

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================

    plt.figure(
        figsize=(12, 5),
        dpi=120
    )

    # Received PSD
    plt.plot(
        freqs / 1e6,
        power_db_norm,
        label="Received PSD",
        lw=0.9
    )

    # CA-CFAR threshold
    plt.plot(
        freqs / 1e6,
        threshold_db_norm,
        '--',
        label="CA-CFAR Threshold",
        color='#ff7f0e',
        lw=1.2
    )

    # Detected interference bins
    if interference_bins:

        plt.scatter(
            freqs[interference_bins] / 1e6,
            power_db_norm[interference_bins],
            color='#1f77b4',
            s=20,
            zorder=5,
            label="Detected interference bins"
        )

    # Protected desired band
    plt.axvspan(
        (
            PROTECTED_CENTER
            - PROTECTED_BANDWIDTH / 2
        ) / 1e6,

        (
            PROTECTED_CENTER
            + PROTECTED_BANDWIDTH / 2
        ) / 1e6,

        alpha=0.15,
        color='#17becf',
        label="Protected desired band"
    )

    # --------------------------------------------------------------------------
    # Annotate detected events
    # --------------------------------------------------------------------------
    for e in event_list:

        plt.annotate(

            f"{e['f_center']:.2f} MHz\n"
            f"SNR {e['snr']:.1f} dB",

            xy=(
                e['f_center'],
                e['peak']
            ),

            xytext=(
                e['f_center'] + 0.3,
                e['peak'] - 8
            ),

            arrowprops=dict(
                arrowstyle="->",
                color="black",
                lw=1
            )
        )

    # --------------------------------------------------------------------------
    # Plot formatting
    # --------------------------------------------------------------------------

    plt.title(
        "Phase 2: Automatic RF Interference Detection "
        "using Linear-Power CA-CFAR"
    )

    plt.xlabel(
        "Frequency (MHz)"
    )

    plt.ylabel(
        "Normalized Power (dB)"
    )

    plt.ylim(
        [-110, 5]
    )

    plt.xlim(
        [
            -FS / (2e6),
            FS / (2e6)
        ]
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.legend(
        loc="upper left"
    )

    plt.tight_layout()

    plt.show()


# ==============================================================================
# PROGRAM ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    main()