================================================================================
                     AUTOMATIC DETECTION OF COMMUNICATION
                    SIGNAL INTERFERENCE USING ADAPTIVE CA-CFAR
                 THRESHOLDING, EVENT EXTRACTION AND RF FEATURE
                                MEASUREMENT
================================================================================

--------------------------------------------------------------------------------
1. PROJECT DETAILS
--------------------------------------------------------------------------------

Project Title:
    Automatic Detection of Communication Signal Interference using
    Adaptive CA-CFAR Thresholding, Event Extraction, and RF Feature
    Measurement

Team Members:
    [R. Neelakesav]        - [2520040028]
    [B. Preetham]        - [2520040044]
    [D.Manish]        - [2520040121]

Guide / Supervisor:
    [Dr.V.Sahithi]
    [Department of Electronics and Communication Engineering]

Institution:
    [KL University, ECE]

Academic Year:
    2025 - 2026

--------------------------------------------------------------------------------
2. SOFTWARE REQUIREMENTS
--------------------------------------------------------------------------------

Operating System:
    - Windows 10 / 11 (tested)
    - Also compatible with Linux (Ubuntu 20.04+) and macOS

Programming Language:
    - Python 3.9 or higher (recommended: 3.10)

IDE:
    - PyCharm Community / Professional (recommended)
    - VS Code, or any Python IDE

Required Python Libraries:
    - numpy            >= 1.24
    - pandas           >= 2.0
    - scipy            >= 1.10
    - matplotlib       >= 3.7
    - scikit-learn     >= 1.3
    - streamlit        >= 1.30
    - plotly           >= 5.18
    - sqlite3          (built into Python, no install required)

Optional Libraries (extend functionality):
    - xgboost          >= 2.0    (Phase 10 - XGBoost classifier)
    - tensorflow       >= 2.15   (Phase 13 - 1D-CNN deep learning)
    - pyrtlsdr         >= 0.2.9  (Phase 11 - real RTL-SDR hardware)

If any of the optional libraries are not installed, the dashboard will
still run — the corresponding features will simply be disabled.

--------------------------------------------------------------------------------
3. INSTALLATION INSTRUCTIONS
--------------------------------------------------------------------------------

Step 1: Open a terminal (PyCharm terminal, PowerShell, or CMD).

Step 2: Navigate to the project folder:
        cd SP

Step 3: (Recommended) Create a virtual environment:
        python -m venv .venv
        .venv\Scripts\activate          (Windows)
        source .venv/bin/activate       (Linux / macOS)

Step 4: Install all required libraries:
        pip install -r requirements.txt

Step 5: Install optional libraries (recommended for full feature set):
        pip install xgboost pyrtlsdr tensorflow

Note: TensorFlow is a ~500 MB download and may take several minutes.

--------------------------------------------------------------------------------
4. PURPOSE OF EACH FILE
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
FILE 1: rf_pipeline.py
--------------------------------------------------------------------------------
PURPOSE:
    The main signal-processing and machine learning pipeline.
    Contains Phases 1-6 and 8-12 of the project.

MAJOR FUNCTIONS:
    Phase 1-2:
        load_or_synthesize_iq()   - Load IQ samples from CSV or simulate
        simulate_frame()          - Generate synthetic jammers (CW, Chirp,
                                    WidebandNoise, Burst, FHSS)
        compute_linear_psd()      - FFT + Hann window + PSD (dB)
        linear_ca_cfar()          - Cell-Averaging CFAR detector
        extract_events()          - Cluster adjacent bins into events

    Phase 3:
        evaluate_pd_pfa()         - Probability of Detection / False Alarm
        snr_sweep_pd_pfa()        - Monte-Carlo SNR sweep + ROC curve

    Phase 4:
        extract_features_12d()    - Extract 12 RF features per event
        build_event_dataset()     - Build labeled training dataset

    Phase 5:
        train_classifier()        - Train Random Forest classifier
        classify_events()         - Predict jammer label for each event

    Phase 6:
        KalmanTrack (class)       - Kalman filter track object
        track_frames()            - Multi-frame association + tracking

    Phase 8:
        os_cfar()                 - Ordered-Statistic CFAR
        vi_cfar()                 - Variability Index CFAR
        cfar_2d()                 - 2D CFAR on time-frequency grid

    Phase 9:
        compute_linear_psd_v2()   - Blackman-Harris window + median filter
        clean_detection_mask()    - Morphological cleanup of detections

    Phase 10:
        train_xgboost()           - Train XGBoost classifier
        load_xgboost()            - Load saved XGBoost model
        classify_events_xgb()     - XGBoost classification + unknown flag

    Phase 11:
        RTLSDRIQSource (class)    - Wrapper for real RTL-SDR hardware

    Phase 12:
        init_event_database()     - Create SQLite event log database
        log_events_to_db()        - Insert new events into DB
        query_recent_events()     - Fetch recent events for display
        check_protected_band_intrusion() - Detect CCI in protected band

INPUTS:
    - Optional CSV file (logged_data.csv) with columns I and Q
    - OR no input (falls back to synthetic simulation)

OUTPUTS:
    - rf_classifier.pkl           (trained Random Forest model)
    - rf_scaler.pkl               (fitted StandardScaler)
    - rf_xgb_model.pkl            (trained XGBoost model, if run)
    - rf_xgb_scaler.pkl           (XGBoost scaler)
    - rf_xgb_encoder.pkl          (label encoder for XGBoost)
    - rf_unknown_detector.pkl     (Isolation Forest for unknown jammers)
    - rf_events.db                (SQLite event log)
    - phase3_roc.png              (ROC curve + SNR sweep plot)
    - phase6_tracks.png           (Kalman track visualization)

--------------------------------------------------------------------------------
FILE 2: cnn_classifier.py
--------------------------------------------------------------------------------
PURPOSE:
    Phase 13 - Deep learning classifier using a 1D-CNN.
    Classifies the full PSD frame (not individual events) using
    automatic feature learning.

MAJOR FUNCTIONS:
    preprocess_psd_for_cnn()  - Crop/pad PSD to fixed size, normalize
    build_cnn()               - Define the 1D-CNN architecture
    build_cnn_dataset()       - Generate labeled PSD frames
    train_cnn()               - Train and save the CNN
    load_cnn()                - Load saved CNN + label encoder
    classify_psd_cnn()        - Predict label + confidence for one frame

INPUTS:
    - Simulated PSD frames (from rf_pipeline)

OUTPUTS:
    - rf_cnn.keras                (trained CNN model)
    - rf_cnn_encoder.pkl          (label encoder)

REQUIRES:
    - TensorFlow (pip install tensorflow)
    - If TensorFlow is not installed, this file is skipped automatically.

--------------------------------------------------------------------------------
FILE 3: dashboard.py
--------------------------------------------------------------------------------
PURPOSE:
    Phase 7 - Real-time Streamlit dashboard.
    The MAIN INTERACTIVE UI. Imports everything from rf_pipeline.py
    and cnn_classifier.py and runs the full pipeline in a live loop.

MAJOR SECTIONS:
    - Sidebar configuration (CFAR mode, classifier, jammer count, etc.)
    - Live waterfall plot (spectrum history)
    - PSD + CFAR threshold overlay
    - KPI metrics (Events, Pd, Pfa, Active tracks)
    - Detected interference events table
    - Kalman track history plot
    - Event heatmap across time
    - SQLite event log table
    - CCI alert banner (protected band intrusion)

INPUTS:
    - Simulated IQ frames (default)
    - OR live RTL-SDR samples (if hardware connected)

OUTPUTS (on screen):
    - Interactive dashboard at http://localhost:8501
    - All charts and tables update in real time

--------------------------------------------------------------------------------
FILE 4: requirements.txt
--------------------------------------------------------------------------------
PURPOSE:
    List of required Python packages for pip installation.

CONTENT:
    numpy>=1.24
    pandas>=2.0
    scipy>=1.10
    matplotlib>=3.7
    scikit-learn>=1.3
    streamlit>=1.30
    plotly>=5.18

--------------------------------------------------------------------------------
FILE 5: logged_data.csv (OPTIONAL)
--------------------------------------------------------------------------------
PURPOSE:
    Real-world IQ data (if available).
    If present, the pipeline can process it instead of simulation.

FORMAT OPTIONS:
    Option A: Two columns  ->  I, Q
    Option B: First two numerical columns are automatically treated as I, Q
    Option C: Single complex-valued column (e.g., "1.2+3.4j")

--------------------------------------------------------------------------------
5. MAIN FILE TO RUN AND EXECUTION SEQUENCE
--------------------------------------------------------------------------------

MAIN FILE TO RUN:
    dashboard.py   (launched via Streamlit)

EXECUTION SEQUENCE (First-Time Setup):

    Step 1: Open terminal and navigate to project folder:
            cd SP

    Step 2: Install required libraries (one-time only):
            pip install -r requirements.txt

    Step 3: (Optional) Install optional libraries:
            pip install xgboost pyrtlsdr tensorflow

    Step 4: (Optional) Pre-train all classifiers to speed up the
            first dashboard load:
            python rf_pipeline.py
            python cnn_classifier.py

    Step 5: Launch the main dashboard:
            streamlit run dashboard.py

    Step 6: Open browser to http://localhost:8501
            (Streamlit opens it automatically in most cases.)

    Step 7: To STOP the dashboard, press Ctrl + C in the terminal.

NOTE:
    - rf_pipeline.py and cnn_classifier.py do NOT need to be run
      manually for the dashboard to work.
    - The dashboard automatically trains classifiers on first use
      if the saved .pkl / .keras files are missing.
    - Running them manually just pre-trains and caches the models
      so the dashboard loads faster.

--------------------------------------------------------------------------------
6. SETTINGS AND FILE PATHS TO CONFIGURE
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
In rf_pipeline.py (top of file - CONFIG section):
--------------------------------------------------------------------------------
    FS = 20.0e6                    Sampling rate (20 MHz)
    N_FFT = 8192                   FFT block size
    NUM_GUARD = 4                  CFAR guard cells
    NUM_REF = 20                   CFAR reference cells
    CFAR_OFFSET_DB = 15.0          CFAR threshold offset in dB
    PROTECTED_CENTER = 2.0e6       Center of protected band (+2 MHz)
    PROTECTED_BANDWIDTH = 0.30e6   Width of protected band
    GROUND_TRUTH_JAMMERS_MHZ = [-3.5, 6.0]   Ground truth for Pd/Pfa
    CLASSES = ["CW", "Chirp", "WidebandNoise", "Burst", "FHSS"]

Model file paths (do not change unless needed):
    MODEL_PATH         = "rf_classifier.pkl"
    SCALER_PATH        = "rf_scaler.pkl"
    XGB_MODEL_PATH     = "rf_xgb_model.pkl"
    UNKNOWN_MODEL_PATH = "rf_unknown_detector.pkl"
    DB_PATH            = "rf_events.db"

--------------------------------------------------------------------------------
In cnn_classifier.py:
--------------------------------------------------------------------------------
    CNN_MODEL_PATH     = "rf_cnn.keras"
    CNN_ENCODER_PATH   = "rf_cnn_encoder.pkl"
    CNN_SIZE           = 1024   (PSD window length fed to CNN)

--------------------------------------------------------------------------------
In dashboard.py (sidebar - adjustable at runtime):
--------------------------------------------------------------------------------
    Update rate (s)        - How often the dashboard refreshes
    CFAR offset (dB)       - Detection threshold
    Jammer amplitude       - Signal strength of simulated jammers
    Number of jammers      - 1 to 10 simultaneous interferers
    Frames in waterfall    - History depth
    CFAR Mode              - CA-CFAR / OS-CFAR / VI-CFAR
    Classifier             - Random Forest / XGBoost / 1D-CNN
    Data Source            - Simulation / Live RTL-SDR

--------------------------------------------------------------------------------
CSV DATASET PATH (if using real data):
--------------------------------------------------------------------------------
    Default path:  "logged_data.csv"  (relative to SP folder)
    Change in dashboard.py or rf_pipeline.py if using a different file.

--------------------------------------------------------------------------------
7. EXPECTED OUTPUTS
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
A. Console Output (when running rf_pipeline.py):
--------------------------------------------------------------------------------
    - Phase 3 SNR sweep table (Pd and Pfa at each SNR level)
    - Classification report (precision, recall, F1-score)
    - Feature importance ranking for Random Forest
    - Phase 6 Kalman tracking summary

--------------------------------------------------------------------------------
B. Saved Files (in SP folder):
--------------------------------------------------------------------------------
    phase3_roc.png               ROC curve and Pd/Pfa vs SNR plot
    phase6_tracks.png            Track evolution across frames
    rf_classifier.pkl            Trained Random Forest
    rf_scaler.pkl                Feature scaler
    rf_xgb_model.pkl             Trained XGBoost (if used)
    rf_unknown_detector.pkl      Isolation Forest (if used)
    rf_cnn.keras                 Trained 1D-CNN (if used)
    rf_events.db                 SQLite event log database

--------------------------------------------------------------------------------
C. Dashboard Output (http://localhost:8501):
--------------------------------------------------------------------------------
    - CCI alert banner (green = clear, red = intrusion detected)
    - Waterfall plot showing spectrum history
    - Power Spectrum with CFAR threshold overlay
    - 4 KPI tiles: Events, Pd, Pfa, Active Tracks
    - Detected interference events table (Label, Confidence,
      Fc, BW, Peak, SNR)
    - Kalman track history plot
    - Event heatmap across time
    - Recent event log (SQLite)

--------------------------------------------------------------------------------
D. SQLite Database (rf_events.db):
--------------------------------------------------------------------------------
    Table: events
    Columns:
        id                    Auto-increment primary key
        timestamp             ISO format timestamp
        f_center_mhz          Detected center frequency
        bandwidth_mhz         Detected bandwidth
        peak_db               Peak power
        snr_db                Signal-to-noise ratio
        label                 Classified jammer type
        confidence            ML confidence score (0-1)
        in_protected_band    1 if inside protected band, 0 otherwise

--------------------------------------------------------------------------------
8. KNOWN LIMITATIONS AND EXECUTION ISSUES
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
A. Simulation-Based Training
--------------------------------------------------------------------------------
    - The ML classifiers are trained on SYNTHETIC data generated by
      simulate_frame(). They have not been trained on real-world RF
      captures unless a labeled real dataset is provided.
    - Classification accuracy on real signals may be lower than the
      accuracy reported in the synthetic classification report.

--------------------------------------------------------------------------------
B. Pd / Pfa Evaluation Uses Fixed Ground Truth
--------------------------------------------------------------------------------
    - The evaluation assumes jammers are always at [-3.5, 6.0] MHz.
    - For FHSS signals that hop outside this range, the strict 0.15 MHz
      tolerance causes false "misses", lowering the reported Pd.
    - This is expected behavior, not a bug.

--------------------------------------------------------------------------------
C. Streamlit Duplicate Element Error
--------------------------------------------------------------------------------
    - If using an older version of dashboard.py (before the fix), you
      may see "StreamlitDuplicateElementId".
    - Fix: every plotly_chart() call now uses a unique key=f"..._{frame_i}".

--------------------------------------------------------------------------------
D. TensorFlow Installation on Windows
--------------------------------------------------------------------------------
    - TensorFlow requires Python 3.9-3.11 on Windows.
    - If installation fails, try:
        pip install tensorflow==2.15.0
    - If it still fails, skip Phase 13 - the dashboard works without it.

--------------------------------------------------------------------------------
E. RTL-SDR Driver Requirements
--------------------------------------------------------------------------------
    - The pyrtlsdr library requires the librtlsdr driver installed on
      the system.
    - On Windows, install the Zadig driver for the RTL-SDR USB device.
    - If the SDR is not connected, the dashboard automatically falls
      back to simulation mode (no crash).

--------------------------------------------------------------------------------
F. Performance Notes
--------------------------------------------------------------------------------
    - The CFAR detectors (CA / OS / VI) use Python loops over every FFT
      bin. On low-end machines with update_rate < 0.2s, the dashboard
      may lag.
    - Solution: increase update_rate to 0.5s or 1.0s.
    - 2D-CFAR is not used in real-time mode (too slow) — it is only
      available as an offline function.

--------------------------------------------------------------------------------
G. SQLite Database Growth
--------------------------------------------------------------------------------
    - The rf_events.db file grows indefinitely as events are logged.
    - To reset, simply delete rf_events.db and restart the dashboard.
    - To keep it small, disable "Log events to SQLite" in the sidebar.

--------------------------------------------------------------------------------
H. Windowing Trade-offs
--------------------------------------------------------------------------------
    - Blackman-Harris window reduces side-lobes but slightly widens the
      main lobe. This can merge very close jammers into one detection.
    - Turn off "Blackman-Harris window" in the sidebar if fine resolution
      is required.

--------------------------------------------------------------------------------
I. Class Imbalance in Training
--------------------------------------------------------------------------------
    - Each class has equal samples (n_per_class), which is balanced.
    - However, real-world jammers may appear at different rates, so the
      classifier is not tuned for rare-event detection.

--------------------------------------------------------------------------------
9. TROUBLESHOOTING QUICK REFERENCE
--------------------------------------------------------------------------------

Problem:  "No such file or directory: 'requirements.txt'"
Fix:      You forgot to run "cd SP" first.

Problem:  "ModuleNotFoundError: No module named 'streamlit'"
Fix:      Run "pip install -r requirements.txt"

Problem:  Streamlit asks for an email on first launch
Fix:      Press Enter (leave blank), then type "y" if prompted.

Problem:  "StreamlitDuplicateElementId"
Fix:      Every plotly_chart must have a unique key= argument.
          This is already fixed in the latest dashboard.py.

Problem:  Dashboard freezes or lags
Fix:      Increase "Update rate" in the sidebar (e.g., 0.5s or 1.0s).
          Reduce "Frames in waterfall".

Problem:  "Address already in use"
Fix:      Another Streamlit instance is running. Press Ctrl+C in the
          old terminal, or kill the process, then retry.

Problem:  No tracks appear in the track plot
Fix:      Wait ~5 seconds. Tracks need 2+ detections to display.
          Check "Enable Kalman tracking" in the sidebar.

--------------------------------------------------------------------------------
10. ACKNOWLEDGEMENTS
--------------------------------------------------------------------------------

    - Base paper reference: [Insert your base paper citation here]
    - Simulation framework inspired by standard RF signal processing
      practices (CFAR, Kalman filtering).
    - Libraries used: NumPy, SciPy, scikit-learn, Streamlit, Plotly,
      TensorFlow, XGBoost.

================================================================================
                            END OF README
================================================================================