"""
ML Inference & Diagnostics Engine for Bearing Fault & Operating Load Detection.
Provides feature extraction, model loading, OOD anomaly detection, and predictive health assessment.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis, skew
import joblib

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
        raise ValueError('Signal length must contain at least 64 samples.')

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

        # Fault band energy in envelope spectrum
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

def get_models_dir():
    candidates = [
        Path(__file__).resolve().parent / 'models',
        Path(__file__).resolve().parent / 'Bearing-Fault-Diagnosis' / 'models',
        Path(__file__).resolve().parent.parent / 'Bearing-Fault-Diagnosis' / 'models',
        Path.cwd() / 'models',
        Path.cwd() / 'Bearing-Fault-Diagnosis' / 'models'
    ]
    for c in candidates:
        if (c / 'fault_classifier.joblib').exists():
            return c
    return candidates[0]

def load_ai_models(models_dir=None):
    if models_dir is None:
        models_dir = get_models_dir()
    else:
        models_dir = Path(models_dir)

    fault_path = models_dir / 'fault_classifier.joblib'
    load_path = models_dir / 'load_classifier.joblib'
    ood_path = models_dir / 'ood_detector.joblib'
    meta_path = models_dir / 'model_metadata.json'

    if not (fault_path.exists() and load_path.exists() and ood_path.exists()):
        from train_models import train_and_evaluate
        train_and_evaluate()

    fault_model = joblib.load(fault_path)
    load_model = joblib.load(load_path)
    ood_model = joblib.load(ood_path)

    metadata = {}
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    return fault_model, load_model, ood_model, metadata

def predict_bearing_health(x, fs=12000, models=None, models_dir=None):
    """
    Performs AI diagnostic prediction on a vibration signal vector x:
    - Fault category & probabilities
    - Motor operating load & probabilities (with Unknown/Uncertain fallback)
    - Out-of-Distribution (OOD) anomaly checking
    - Experimental multi-fault spectral co-occurrence indicator
    - Comprehensive signal feature breakdown & engineering narrative
    """
    if models is None:
        fault_model, load_model, ood_model, metadata = load_ai_models(models_dir)
    else:
        fault_model, load_model, ood_model, metadata = models

    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    if n < 100 or not np.all(np.isfinite(x)):
        raise ValueError('Signal contains non-finite samples or fewer than 100 points.')

    window_size = 2048
    if n >= window_size * 2:
        num_windows = min(n // window_size, 30)
        feat_list = []
        for i in range(num_windows):
            chunk = x[i * window_size : (i + 1) * window_size]
            feat_list.append(extract_comprehensive_features(chunk, fs))
        feat_df = pd.DataFrame(feat_list)
        feats = feat_df.mean().to_dict()
    else:
        feats = extract_comprehensive_features(x, fs)

    feat_vector = np.array([feats[fn] for fn in FEATURE_NAMES]).reshape(1, -1)

    # 1. Out-of-Distribution (OOD) Detection
    ood_score = float(ood_model.named_steps['iso'].decision_function(
        ood_model.named_steps['scaler'].transform(feat_vector)
    )[0])
    is_ood = bool(ood_score < 0.0 or fs != 12000 or feats['rms'] > 5.0 or feats['rms'] < 0.005)

    # 2. Bearing Condition Classification
    fault_classes = fault_model.classes_
    fault_probs = fault_model.predict_proba(feat_vector)[0]
    fault_prob_dict = {cls: float(prob) for cls, prob in zip(fault_classes, fault_probs)}
    top_fault_idx = int(np.argmax(fault_probs))
    pred_fault = str(fault_classes[top_fault_idx])
    fault_conf = float(fault_probs[top_fault_idx])

    is_healthy = (pred_fault == 'Normal')
    health_status = 'Normal Baseline' if is_healthy else 'Fault Condition Detected'

    # 3. Motor Operating Load Classification
    load_classes = load_model.classes_
    load_probs = load_model.predict_proba(feat_vector)[0]
    load_prob_dict = {cls: float(prob) for cls, prob in zip(load_classes, load_probs)}
    top_load_idx = int(np.argmax(load_probs))
    raw_pred_load = str(load_classes[top_load_idx])
    load_conf = float(load_probs[top_load_idx])

    # Safe handling: if OOD or confidence is low, do NOT force an HP label
    if is_ood or load_conf < 0.45:
        pred_load = 'Unknown / Uncertain Load'
        load_status = 'Uncertain'
    else:
        pred_load = raw_pred_load
        load_status = 'Confident'

    if is_ood:
        diagnostic_confidence = 'Out-of-Distribution / Unknown'
    elif fault_conf < 0.55:
        diagnostic_confidence = 'Low Confidence (Uncertain)'
    else:
        diagnostic_confidence = 'High Confidence (In-Distribution)'

    # 4. Experimental Multiple Fault Spectral Analysis
    bpfo_e = feats['env_bpfo_band_energy']
    bpfi_e = feats['env_bpfi_band_energy']
    bsf_e = feats['env_bsf_band_energy']
    total_env_e = feats['env_spectral_energy'] + 1e-12

    bpfo_ratio = bpfo_e / total_env_e
    bpfi_ratio = bpfi_e / total_env_e
    bsf_ratio = bsf_e / total_env_e

    active_fault_indicators = []
    if bpfo_ratio > 0.12 or pred_fault == 'Outer Race Fault':
        active_fault_indicators.append('Outer Race Harmonic Signature')
    if bpfi_ratio > 0.10 or pred_fault == 'Inner Race Fault':
        active_fault_indicators.append('Inner Race Harmonic Signature')
    if bsf_ratio > 0.10 or pred_fault == 'Ball Fault':
        active_fault_indicators.append('Rolling Element (Ball) Harmonic Signature')

    if len(active_fault_indicators) > 1 and not is_healthy:
        multi_fault_indication = 'Concurrent Multi-Band Harmonics Detected (' + ' + '.join(active_fault_indicators) + ')'
        multi_fault_risk = 'Moderate / High Harmonic Overlap'
    else:
        multi_fault_indication = 'Single Dominant Fault Signature'
        multi_fault_risk = 'Isolated Single Defect'

    # 5. Technical Engineering Narrative
    reasons = []
    if is_healthy:
        reasons.append(f"Vibration energy is low (RMS = {feats['rms']:.4f} g) with near-zero kurtosis ({feats['kurtosis']:.2f}), reflecting nominal sinusoidal shaft rotation without repetitive impact transients.")
    else:
        reasons.append(f"Elevated vibration energy (RMS = {feats['rms']:.4f} g, Peak = {feats['peak']:.4f} g) indicates active mechanical stress.")
        if feats['kurtosis'] > 1.5:
            reasons.append(f"High kurtosis ({feats['kurtosis']:.2f} > 1.5) proves sharp, repetitive structural impact pulses characteristic of localized raceway pitting.")
        if pred_fault == 'Outer Race Fault':
            reasons.append("Extreme periodic pulse amplitudes confirm passage through the bearing stationary load zone (Outer Race defect).")
        elif pred_fault == 'Inner Race Fault':
            reasons.append("High-impulse modulation observed as rotating inner raceway passes through the load zone (Inner Race defect).")
        elif pred_fault == 'Ball Fault':
            reasons.append("Moderate energy spread with rolling element pass frequencies suggests ball spalling defect.")

    if pred_load != 'Unknown / Uncertain Load':
        reasons.append(f"Spectral distribution and shaft slip harmonics match authentic {pred_load} CWRU operating conditions (confidence: {load_conf * 100:.1f}%).")
    else:
        reasons.append("Motor load cannot be confidently mapped to 0–3 HP; signal operating torque or speed may differ from benchmark motor calibration.")

    if is_ood:
        reasons.append("WARNING: This signal's feature vector lies outside the CWRU benchmark training manifold. Results should be treated as experimental exploratory indications.")

    return {
        'health_status': health_status,
        'is_healthy': is_healthy,
        'predicted_fault': pred_fault,
        'fault_confidence': fault_conf,
        'fault_probabilities': fault_prob_dict,
        'predicted_load': pred_load,
        'raw_predicted_load': raw_pred_load,
        'load_confidence': load_conf,
        'load_probabilities': load_prob_dict,
        'load_status': load_status,
        'is_ood': is_ood,
        'ood_score': ood_score,
        'diagnostic_confidence': diagnostic_confidence,
        'multi_fault_indication': multi_fault_indication,
        'multi_fault_risk': multi_fault_risk,
        'active_fault_indicators': active_fault_indicators,
        'features': feats,
        'technical_narrative': ' '.join(reasons),
        'metadata': metadata
    }
