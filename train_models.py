"""
Model Training & Validation Pipeline for Bearing Fault & Operating Load Detection.
Learns from authentic Case Western Reserve University (CWRU) 12 kHz Drive-End recordings.
Produces trained models for:
1. Fault Classifier (Normal, Inner Race Fault, Ball Fault, Outer Race Fault)
2. Motor Load Classifier (0 HP, 1 HP, 2 HP, 3 HP)
3. Out-of-Distribution (OOD) / Anomaly Detector (IsolationForest)
Saves models and comprehensive evaluation metrics to disk.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_predict, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

DATA_DIR = Path(__file__).resolve().parent / 'data'
MODELS_DIR = Path(__file__).resolve().parent / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 16 Authentic CWRU recordings: 4 fault categories across 4 load levels
FILES_META = {
    '97.mat.txt': ('Normal', '0 HP', 'X097_DE_time'),
    '98.mat.txt': ('Normal', '1 HP', 'X098_DE_time'),
    '99.mat.txt': ('Normal', '2 HP', 'X099_DE_time'),
    '100.mat.txt': ('Normal', '3 HP', 'X100_DE_time'),
    '105.mat.txt': ('Inner Race Fault', '0 HP', 'X105_DE_time'),
    '106.mat.txt': ('Inner Race Fault', '1 HP', 'X106_DE_time'),
    '107.mat.txt': ('Inner Race Fault', '2 HP', 'X107_DE_time'),
    '108.mat.txt': ('Inner Race Fault', '3 HP', 'X108_DE_time'),
    '118.mat.txt': ('Ball Fault', '0 HP', 'X118_DE_time'),
    '119.mat.txt': ('Ball Fault', '1 HP', 'X119_DE_time'),
    '120.mat.txt': ('Ball Fault', '2 HP', 'X120_DE_time'),
    '121.mat.txt': ('Ball Fault', '3 HP', 'X121_DE_time'),
    '130.mat.txt': ('Outer Race Fault', '0 HP', 'X130_DE_time'),
    '131.mat.txt': ('Outer Race Fault', '1 HP', 'X131_DE_time'),
    '132.mat.txt': ('Outer Race Fault', '2 HP', 'X132_DE_time'),
    '133.mat.txt': ('Outer Race Fault', '3 HP', 'X133_DE_time'),
}

FS = 12000
WINDOW_SIZE = 2048
MAX_WINDOWS = 60

FEATURE_NAMES = [
    'rms', 'mean', 'std', 'var', 'skewness', 'kurtosis', 'peak', 'ptp',
    'crest_factor', 'shape_factor', 'impulse_factor', 'margin_factor',
    'dominant_freq', 'spectral_centroid', 'spectral_spread', 'spectral_skewness',
    'spectral_kurtosis', 'spectral_energy', 'band_0_100_ratio', 'band_100_500_ratio',
    'band_500_2000_ratio', 'band_2000_5000_ratio', 'env_rms', 'env_peak',
    'env_crest', 'env_kurtosis', 'env_dominant_freq', 'env_spectral_energy',
    'env_bpfo_band_energy', 'env_bpfi_band_energy', 'env_bsf_band_energy'
]

def extract_comprehensive_features(x, fs=12000):
    """Extracts 31 time-domain, frequency-domain, and envelope-domain features from vibration signal."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    if n < 64:
        raise ValueError('Signal sample count is too small for feature extraction.')

    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x * x)))
    mean_val = float(np.mean(x))
    std_val = float(np.std(x))
    var_val = float(np.var(x))
    skew_val = float(skew(x))
    kurt_val = float(kurtosis(x, fisher=True, bias=True))
    peak_val = float(np.max(abs_x))
    ptp_val = float(np.ptp(x))
    mean_abs = float(np.mean(abs_x)) if float(np.mean(abs_x)) > 1e-12 else 1e-12
    mean_sqrt = float(np.mean(np.sqrt(abs_x))) ** 2 if float(np.mean(np.sqrt(abs_x))) > 1e-12 else 1e-12

    crest_factor = peak_val / (rms + 1e-12)
    shape_factor = rms / mean_abs
    impulse_factor = peak_val / mean_abs
    margin_factor = peak_val / mean_sqrt

    # Frequency Domain (0 to 5000 Hz)
    freq = np.fft.rfftfreq(n, 1 / fs)
    win = np.hanning(n)
    mag = 2 * np.abs(np.fft.rfft((x - mean_val) * win)) / max(np.sum(win), 1)
    mask = freq <= 5000
    f_sub, a_sub = freq[mask], mag[mask]

    sum_a = float(np.sum(a_sub)) + 1e-12
    dom_freq = float(f_sub[np.argmax(a_sub)]) if len(a_sub) > 0 else 0.0
    spec_energy = float(np.sum(a_sub ** 2))
    spec_centroid = float(np.sum(f_sub * a_sub) / sum_a)
    spec_spread = float(np.sqrt(np.sum(((f_sub - spec_centroid) ** 2) * a_sub) / sum_a))
    spec_skew = float(np.sum(((f_sub - spec_centroid) ** 3) * a_sub) / ((spec_spread ** 3 + 1e-12) * sum_a))
    spec_kurt = float(np.sum(((f_sub - spec_centroid) ** 4) * a_sub) / ((spec_spread ** 4 + 1e-12) * sum_a))

    b1 = float(np.sum(a_sub[(f_sub >= 0) & (f_sub < 100)] ** 2))
    b2 = float(np.sum(a_sub[(f_sub >= 100) & (f_sub < 500)] ** 2))
    b3 = float(np.sum(a_sub[(f_sub >= 500) & (f_sub < 2000)] ** 2))
    b4 = float(np.sum(a_sub[(f_sub >= 2000) & (f_sub <= 5000)] ** 2))
    tot_b = b1 + b2 + b3 + b4 + 1e-12

    # Envelope Domain (Bandpass 100-5000 Hz + Hilbert)
    try:
        low = max(100.0, fs * 0.08)
        high = min(5000.0, fs * 0.42)
        sos = butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
        filtered = sosfiltfilt(sos, x)
        env = np.abs(hilbert(filtered))
        env_rms = float(np.sqrt(np.mean(env ** 2)))
        env_peak = float(np.max(env))
        env_crest = env_peak / (env_rms + 1e-12)
        env_kurt = float(kurtosis(env, fisher=True, bias=True))

        env_n = len(env)
        env_f = np.fft.rfftfreq(env_n, 1 / fs)
        env_win = np.hanning(env_n)
        env_mag = 2 * np.abs(np.fft.rfft((env - np.mean(env)) * env_win)) / max(np.sum(env_win), 1)
        env_mask = (env_f >= 5) & (env_f <= 800)
        ef_sub, ea_sub = env_f[env_mask], env_mag[env_mask]
        env_dom_freq = float(ef_sub[np.argmax(ea_sub)]) if len(ea_sub) > 0 else 0.0
        env_spec_energy = float(np.sum(ea_sub ** 2))

        # Bearing characteristic fault bands in envelope spectrum
        bpfo_energy = float(np.sum(ea_sub[(ef_sub >= 95) & (ef_sub <= 118)] ** 2))
        bpfi_energy = float(np.sum(ea_sub[(ef_sub >= 148) & (ef_sub <= 172)] ** 2))
        bsf_energy = float(np.sum(ea_sub[(ef_sub >= 128) & (ef_sub <= 148)] ** 2))
    except Exception:
        env_rms, env_peak, env_crest, env_kurt = rms, peak_val, crest_factor, kurt_val
        env_dom_freq, env_spec_energy = dom_freq, spec_energy
        bpfo_energy, bpfi_energy, bsf_energy = 0.0, 0.0, 0.0

    return {
        'rms': rms, 'mean': mean_val, 'std': std_val, 'var': var_val,
        'skewness': skew_val, 'kurtosis': kurt_val, 'peak': peak_val, 'ptp': ptp_val,
        'crest_factor': crest_factor, 'shape_factor': shape_factor,
        'impulse_factor': impulse_factor, 'margin_factor': margin_factor,
        'dominant_freq': dom_freq, 'spectral_centroid': spec_centroid,
        'spectral_spread': spec_spread, 'spectral_skewness': spec_skew,
        'spectral_kurtosis': spec_kurt, 'spectral_energy': spec_energy,
        'band_0_100_ratio': b1 / tot_b, 'band_100_500_ratio': b2 / tot_b,
        'band_500_2000_ratio': b3 / tot_b, 'band_2000_5000_ratio': b4 / tot_b,
        'env_rms': env_rms, 'env_peak': env_peak, 'env_crest': env_crest,
        'env_kurtosis': env_kurt, 'env_dominant_freq': env_dom_freq,
        'env_spectral_energy': env_spec_energy,
        'env_bpfo_band_energy': bpfo_energy,
        'env_bpfi_band_energy': bpfi_energy,
        'env_bsf_band_energy': bsf_energy
    }

def build_dataset():
    records = []
    print(f'Loading {len(FILES_META)} authentic CWRU recordings from: {DATA_DIR}')
    for fn, (fault, load, key) in FILES_META.items():
        fp = DATA_DIR / fn
        if not fp.exists():
            print(f'Warning: {fn} not found in {DATA_DIR}')
            continue
        m = loadmat(str(fp))
        x_full = np.asarray(m[key], dtype=float).reshape(-1)
        n_windows = min(len(x_full) // WINDOW_SIZE, MAX_WINDOWS)
        for i in range(n_windows):
            chunk = x_full[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]
            feats = extract_comprehensive_features(chunk, FS)
            feats['fault_class'] = fault
            feats['load_class'] = load
            feats['file_name'] = fn
            records.append(feats)

    df = pd.DataFrame(records)
    print(f'Constructed training dataset: {len(df)} samples across {df["fault_class"].nunique()} fault classes and {df["load_class"].nunique()} load classes.')
    return df

def train_and_evaluate():
    df = build_dataset()
    X = df[FEATURE_NAMES].values
    y_fault = df['fault_class'].values
    y_load = df['load_class'].values

    # Stratified 5-Fold Cross Validation for Fault Classification
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_fault_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42))
    ])
    fault_cv_preds = cross_val_predict(rf_fault_pipe, X, y_fault, cv=skf)
    fault_acc = float(accuracy_score(y_fault, fault_cv_preds))
    fault_report = classification_report(y_fault, fault_cv_preds, output_dict=True)
    fault_cm = confusion_matrix(y_fault, fault_cv_preds, labels=sorted(list(set(y_fault)))).tolist()

    print(f'\n=== FAULT CLASSIFIER EVALUATION ===')
    print(f'5-Fold Stratified Accuracy: {fault_acc * 100:.2f}%')
    for cls_name, m in fault_report.items():
        if isinstance(m, dict):
            print(f'  {cls_name:18s} Precision: {m["precision"]:.3f} | Recall: {m["recall"]:.3f} | F1: {m["f1-score"]:.3f}')

    # Leave-One-Load-Out validation for Fault Classifier (Cross-Load Generalization)
    logo = LeaveOneGroupOut()
    lolo_preds, lolo_true = [], []
    for train_idx, test_idx in logo.split(X, y_fault, groups=df['load_class'].values):
        rf_fault_pipe.fit(X[train_idx], y_fault[train_idx])
        lolo_preds.extend(rf_fault_pipe.predict(X[test_idx]))
        lolo_true.extend(y_fault[test_idx])
    lolo_acc = float(accuracy_score(lolo_true, lolo_preds))
    print(f'Leave-One-Load-Out Fault Accuracy (Generalization across unseen motor loads): {lolo_acc * 100:.2f}%')

    # Stratified 5-Fold Cross Validation for Operating Load Classification
    rf_load_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42))
    ])
    load_cv_preds = cross_val_predict(rf_load_pipe, X, y_load, cv=skf)
    load_acc = float(accuracy_score(y_load, load_cv_preds))
    load_report = classification_report(y_load, load_cv_preds, output_dict=True)
    load_cm = confusion_matrix(y_load, load_cv_preds, labels=sorted(list(set(y_load)))).tolist()

    print(f'\n=== LOAD CLASSIFIER EVALUATION ===')
    print(f'5-Fold Stratified Accuracy: {load_acc * 100:.2f}%')
    for cls_name, m in load_report.items():
        if isinstance(m, dict):
            print(f'  {cls_name:18s} Precision: {m["precision"]:.3f} | Recall: {m["recall"]:.3f} | F1: {m["f1-score"]:.3f}')

    # Train Final Full Production Models
    print('\nTraining final production models on full CWRU benchmark...')
    rf_fault_pipe.fit(X, y_fault)
    rf_load_pipe.fit(X, y_load)

    # Train Out-of-Distribution (OOD) Detector on In-Distribution Feature Space
    scaler_ood = StandardScaler()
    X_scaled = scaler_ood.fit_transform(X)
    iso_forest = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
    iso_forest.fit(X_scaled)
    ood_pipe = Pipeline([('scaler', scaler_ood), ('iso', iso_forest)])

    # Feature Importances
    fault_clf_rf = rf_fault_pipe.named_steps['clf']
    fault_feat_imp = {fn: round(float(imp), 4) for fn, imp in zip(FEATURE_NAMES, fault_clf_rf.feature_importances_)}
    fault_feat_imp = dict(sorted(fault_feat_imp.items(), key=lambda item: item[1], reverse=True))

    load_clf_rf = rf_load_pipe.named_steps['clf']
    load_feat_imp = {fn: round(float(imp), 4) for fn, imp in zip(FEATURE_NAMES, load_clf_rf.feature_importances_)}
    load_feat_imp = dict(sorted(load_feat_imp.items(), key=lambda item: item[1], reverse=True))

    # Serialize Models
    joblib.dump(rf_fault_pipe, MODELS_DIR / 'fault_classifier.joblib')
    joblib.dump(rf_load_pipe, MODELS_DIR / 'load_classifier.joblib')
    joblib.dump(ood_pipe, MODELS_DIR / 'ood_detector.joblib')

    # Also mirror into Bearing-Fault-Diagnosis-main if present
    mirror_dir = Path(__file__).resolve().parent.parent / 'Bearing-Fault-Diagnosis-main' / 'models'
    if mirror_dir.parent.exists():
        mirror_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf_fault_pipe, mirror_dir / 'fault_classifier.joblib')
        joblib.dump(rf_load_pipe, mirror_dir / 'load_classifier.joblib')
        joblib.dump(ood_pipe, mirror_dir / 'ood_detector.joblib')

    metadata = {
        'training_source': 'CWRU 12 kHz Drive-End Accelerometer Benchmark',
        'sample_rate_hz': FS,
        'window_size': WINDOW_SIZE,
        'total_training_samples': len(df),
        'features': FEATURE_NAMES,
        'fault_classes': sorted(list(set(y_fault))),
        'load_classes': sorted(list(set(y_load))),
        'evaluation': {
            'fault_classification': {
                '5_fold_cv_accuracy': round(fault_acc, 4),
                'leave_one_load_out_accuracy': round(lolo_acc, 4),
                'report': fault_report,
                'confusion_matrix': fault_cm,
                'top_features': dict(list(fault_feat_imp.items())[:8])
            },
            'load_classification': {
                '5_fold_cv_accuracy': round(load_acc, 4),
                'report': load_report,
                'confusion_matrix': load_cm,
                'top_features': dict(list(load_feat_imp.items())[:8])
            }
        },
        'multi_fault_status': {
            'supported': False,
            'reason': 'CWRU Drive-End benchmark contains single-defect bearings only. No simultaneous multi-defect ground truth recordings exist in authentic dataset.'
        }
    }

    with open(MODELS_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    if mirror_dir.exists():
        with open(mirror_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f'Successfully trained and saved all models to: {MODELS_DIR}')
    return metadata

if __name__ == '__main__':
    train_and_evaluate()
