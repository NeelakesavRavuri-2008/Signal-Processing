"""
Phase 13: 1D-CNN classifier for RF interference.
Operates directly on the PSD (no manual feature extraction).
"""
import os
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # silence TF logs
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

CNN_MODEL_PATH = "rf_cnn.h5"
CNN_ENCODER_PATH = "rf_cnn_encoder.pkl"
CNN_SIZE = 1024       # PSD window fed to the CNN


def preprocess_psd_for_cnn(p_db, target_size=CNN_SIZE):
    """Crop or pad the PSD to a fixed size and normalize."""
    n = len(p_db)
    if n >= target_size:
        start = (n - target_size) // 2
        x = p_db[start:start + target_size]
    else:
        x = np.pad(p_db, (0, target_size - n))
    # Normalize to [-1, 1]
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return x.astype(np.float32)


def build_cnn(input_size=CNN_SIZE, n_classes=5):
    """Lightweight 1D-CNN for spectral classification."""
    model = models.Sequential([
        layers.Input(shape=(input_size, 1)),
        layers.Conv1D(32, 7, padding='same', activation='relu'),
        layers.MaxPooling1D(4),
        layers.Conv1D(64, 5, padding='same', activation='relu'),
        layers.MaxPooling1D(4),
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_cnn_dataset(n_per_class=200, seed=0):
    """Generate PSD frames labeled by jammer type."""
    from rf_pipeline import simulate_frame, compute_linear_psd, \
        N_FFT, FS, CLASSES
    rng = np.random.default_rng(seed)
    X, y = [], []

    for cls in CLASSES:
        print(f"  Generating {n_per_class} '{cls}' PSDs ...")
        for _ in range(n_per_class):
            f0 = rng.uniform(-8e6, 8e6)
            amp = rng.uniform(0.3, 1.2)
            bw = {"CW": 1e5, "Chirp": 2e6, "WidebandNoise": 3e6,
                  "Burst": 1e5, "FHSS": 2e6}[cls]
            j = {"type": cls, "f": f0, "bw": bw, "amp": amp}
            iq = simulate_frame(N_FFT, FS, jammers=[j],
                                seed=int(rng.integers(0, 1e6)))
            _, _, p_db = compute_linear_psd(iq, FS)
            X.append(preprocess_psd_for_cnn(p_db))
            y.append(cls)
    return np.array(X)[..., np.newaxis], np.array(y)


def train_cnn(X, y, epochs=15, batch_size=32, save=True):
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = build_cnn(input_size=X.shape[1], n_classes=len(le.classes_))
    model.summary()
    history = model.fit(Xtr, ytr, validation_data=(Xte, yte),
                        epochs=epochs, batch_size=batch_size, verbose=1)

    if save:
        model.save(CNN_MODEL_PATH)
        with open(CNN_ENCODER_PATH, 'wb') as f:
            pickle.dump(le, f)
        print(f"Saved: {CNN_MODEL_PATH}, {CNN_ENCODER_PATH}")

    return model, le, history


def load_cnn():
    if not os.path.exists(CNN_MODEL_PATH):
        return None, None
    model = tf.keras.models.load_model(CNN_MODEL_PATH)
    with open(CNN_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    return model, le


def classify_psd_cnn(p_db, model, le):
    """Classify a single PSD frame with the CNN."""
    x = preprocess_psd_for_cnn(p_db)[np.newaxis, :, np.newaxis]
    proba = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(proba))
    return le.inverse_transform([idx])[0], float(proba[idx])