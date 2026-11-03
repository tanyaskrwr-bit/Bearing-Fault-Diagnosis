"""
Bearing Fault Diagnosis — Advanced Expo Dashboard.
Industrial-grade Condition Monitoring & Vibration Signal Intelligence Suite.
Designed for Engineering Expo & Research Demonstration.

Install: pip install streamlit numpy pandas matplotlib scipy
Run: python -m streamlit run app_expo_advanced_v2.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.io import loadmat
from io import BytesIO
from scipy.signal import spectrogram, find_peaks, butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis
import ml_engine as mle

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
FS = 12000
DATA = Path(__file__).resolve().parent / 'data'
NAMES = ['Normal', 'Inner Race Fault', 'Ball Fault', 'Outer Race Fault']

# High-contrast industrial diagnostic color palette
COLORS = {
    'Normal': '#10B981',           # Vibrant Emerald - Healthy Baseline
    'Inner Race Fault': '#F59E0B', # Vibrant Amber - Impulsive Bearing Impact
    'Ball Fault': '#06B6D4',       # Electric Cyan - Rolling Element Defect
    'Outer Race Fault': '#EF4444'  # High-Alert Coral - Outer Raceway Defect
}

FILES = {
    '0 HP': {
        'Normal': ('97.mat.txt', 'X097_DE_time'),
        'Inner Race Fault': ('105.mat.txt', 'X105_DE_time'),
        'Ball Fault': ('118.mat.txt', 'X118_DE_time'),
        'Outer Race Fault': ('130.mat.txt', 'X130_DE_time')
    },
    '1 HP': {
        'Normal': ('98.mat.txt', 'X098_DE_time'),
        'Inner Race Fault': ('106.mat.txt', 'X106_DE_time'),
        'Ball Fault': ('119.mat.txt', 'X119_DE_time'),
        'Outer Race Fault': ('131.mat.txt', 'X131_DE_time')
    },
    '2 HP': {
        'Normal': ('99.mat.txt', 'X099_DE_time'),
        'Inner Race Fault': ('107.mat.txt', 'X107_DE_time'),
        'Ball Fault': ('120.mat.txt', 'X120_DE_time'),
        'Outer Race Fault': ('132.mat.txt', 'X132_DE_time')
    },
    '3 HP': {
        'Normal': ('100.mat.txt', 'X100_DE_time'),
        'Inner Race Fault': ('108.mat.txt', 'X108_DE_time'),
        'Ball Fault': ('121.mat.txt', 'X121_DE_time'),
        'Outer Race Fault': ('133.mat.txt', 'X133_DE_time')
    },
}

@st.cache_resource(show_spinner=False)
def get_ai_models():
    """Loads and caches the trained AI classification and OOD models."""
    return mle.load_ai_models()

# ==============================================================================
# STREAMLIT PAGE CONFIG & ADVANCED INDUSTRIAL HUD THEME (CSS)
# ==============================================================================
st.set_page_config(
    page_title='BearingIQ Pro | Industrial Vibration Diagnostics',
    page_icon='⚙️',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-base: #070C16;
    --bg-surface: #0B1322;
    --bg-card: #0F1A2E;
    --bg-card-hover: #14223A;
    --border-subtle: rgba(0, 229, 255, 0.14);
    --border-strong: rgba(0, 229, 255, 0.35);
    --cyan-accent: #00E5FF;
    --teal-accent: #00B4D8;
    --emerald-accent: #10B981;
    --amber-accent: #F59E0B;
    --coral-accent: #EF4444;
    --text-main: #F1F6FD;
    --text-muted: #8EA5C0;
    --text-dim: #5A7391;
    color-scheme: dark;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-main);
}

.stApp {
    background: radial-gradient(ellipse at 15% 0%, #122238 0%, #070C16 55%, #040810 100%);
    color: var(--text-main);
}

.block-container {
    padding: 1.5rem clamp(1.2rem, 2.5vw, 2.8rem) 3.5rem;
    max-width: 1680px;
}

[data-testid="stHeader"] {
    background: rgba(7, 12, 22, 0.82);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border-subtle);
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #08101E 0%, #060A14 100%);
    border-right: 1px solid var(--border-subtle);
}
[data-testid="stSidebar"] hr {
    border-color: rgba(0, 229, 255, 0.12);
    margin: 1.1rem 0;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-muted);
    font-size: 0.88rem;
}

/* Sidebar Brand Block */
.sidebar-brand-card {
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.1) 0%, rgba(0, 180, 216, 0.04) 100%);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
.sidebar-brand-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--cyan-accent), var(--teal-accent), transparent);
}
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #FFFFFF;
}
.sidebar-logo span {
    color: var(--cyan-accent);
}
.sidebar-sub {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Sidebar Section Headers */
.sidebar-section-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--cyan-accent);
    margin: 18px 0 8px 2px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Sidebar Telemetry Pill */
.sidebar-telemetry-box {
    background: #091220;
    border: 1px solid rgba(0, 229, 255, 0.16);
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 20px;
}
.sidebar-telemetry-title {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
}
.sidebar-telemetry-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.76rem;
    padding: 3px 0;
    border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
}
.sidebar-telemetry-row:last-child {
    border-bottom: none;
}
.sidebar-telemetry-row span { color: var(--text-muted); }
.sidebar-telemetry-row b { color: #E0EEFD; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; }

/* Form Controls & Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #0B1424 !important;
    border: 1px solid #1C3352 !important;
    border-radius: 9px !important;
    color: #F1F6FD !important;
    transition: all 0.2s ease !important;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--cyan-accent) !important;
    box-shadow: 0 0 12px rgba(0, 229, 255, 0.22) !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #F1F6FD !important;
    -webkit-text-fill-color: #F1F6FD !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] svg { fill: var(--cyan-accent) !important; }

ul[role="listbox"], div[role="listbox"] {
    background: #0C1729 !important;
    border: 1px solid #1F3B60 !important;
    border-radius: 10px !important;
    box-shadow: 0 12px 30px rgba(0,0,0,0.6) !important;
}
li[role="option"] {
    color: #E2EEFD !important;
    font-size: 0.88rem !important;
    padding: 9px 14px !important;
}
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background: #142B4C !important;
    color: var(--cyan-accent) !important;
}

/* Radio buttons styled as modern status cards */
div[data-testid="stRadio"] > div {
    gap: 8px !important;
}
div[data-testid="stRadio"] label {
    background: rgba(11, 20, 36, 0.7) !important;
    border: 1px solid #172D4A !important;
    padding: 7px 12px !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stRadio"] label:hover {
    border-color: var(--cyan-accent) !important;
    background: rgba(0, 229, 255, 0.08) !important;
}

/* KPI Metric Cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0F1C31 0%, #091220 100%) !important;
    border: 1px solid rgba(0, 229, 255, 0.18) !important;
    border-top: 3px solid var(--cyan-accent) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
    min-height: 110px !important;
    transition: transform 0.15s ease, border-color 0.15s ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: var(--cyan-accent) !important;
    box-shadow: 0 10px 28px rgba(0, 229, 255, 0.15) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: clamp(1.25rem, 1.85vw, 1.85rem) !important;
    font-weight: 800 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.03em !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] {
    color: var(--cyan-accent) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"] svg { fill: var(--cyan-accent) !important; }

/* Main Dashboard Header HUD */
.hero-hud {
    background: linear-gradient(125deg, #0A1629 0%, #0E2442 55%, #083344 100%);
    border: 1px solid var(--border-strong);
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 22px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
    position: relative;
    overflow: hidden;
}
.hero-hud::after {
    content: "";
    position: absolute;
    right: -40px;
    top: -90px;
    width: 280px;
    height: 280px;
    border: 1px solid rgba(0, 229, 255, 0.22);
    border-radius: 50%;
    box-shadow: 0 0 0 20px rgba(0, 229, 255, 0.04), 0 0 0 45px rgba(0, 229, 255, 0.02);
    pointer-events: none;
}
.hero-top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 12px;
    position: relative;
    z-index: 1;
}
.hero-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.38);
    color: #6EE7B7;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.hero-pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 8px #10B981;
    animation: livePulse 2s infinite ease-in-out;
}
@keyframes livePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.3); }
}
.hero-tag-group {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.hero-pill-badge {
    background: rgba(0, 229, 255, 0.08);
    border: 1px solid rgba(0, 229, 255, 0.25);
    color: #94E7F7;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
}
.hero-pill-badge.accent {
    background: rgba(0, 229, 255, 0.18);
    border-color: var(--cyan-accent);
    color: #FFFFFF;
    font-weight: 700;
}
.hero-main-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 24px;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}
.hero-title-group h1 {
    margin: 0;
    font-size: clamp(1.6rem, 2.7vw, 2.35rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-title-group h1 span.cyan-glow {
    color: var(--cyan-accent);
    text-shadow: 0 0 20px rgba(0, 229, 255, 0.45);
}
.hero-subtitle {
    margin: 8px 0 0;
    font-size: 0.92rem;
    color: #A5C4E3;
    font-weight: 500;
    letter-spacing: 0.02em;
    max-width: 820px;
}
.hero-spec-chips {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.spec-chip {
    background: rgba(7, 14, 25, 0.72);
    border: 1px solid rgba(0, 229, 255, 0.22);
    padding: 8px 14px;
    border-radius: 9px;
    min-width: 125px;
}
.spec-chip .lbl {
    display: block;
    font-size: 0.66rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
}
.spec-chip .val {
    display: block;
    font-size: 0.88rem;
    font-weight: 700;
    color: #FFFFFF;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}

/* Module Section Header Banner */
.module-hud-banner {
    background: linear-gradient(90deg, rgba(14, 26, 46, 0.9) 0%, rgba(9, 17, 30, 0.75) 100%);
    border-left: 4px solid var(--cyan-accent);
    border-radius: 0 10px 10px 0;
    padding: 12px 18px;
    margin: 10px 0 20px 0;
}
.module-breadcrumb {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--cyan-accent);
    margin-bottom: 3px;
}
.module-hud-banner h2 {
    margin: 0;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #FFFFFF;
}
.module-desc {
    font-size: 0.84rem;
    color: var(--text-muted);
    margin: 4px 0 0 0;
}

/* Buttons & Download buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #007799 0%, #00A8CC 50%, #00C896 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--cyan-accent) !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    padding: 0.52rem 1.25rem !important;
    box-shadow: 0 4px 16px rgba(0, 180, 216, 0.25) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0099CC 0%, #00D2FF 100%) !important;
    border-color: #80EFFF !important;
    box-shadow: 0 6px 24px rgba(0, 229, 255, 0.45) !important;
    transform: translateY(-2px) !important;
}

/* Sliders */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--cyan-accent) !important;
    border-color: var(--cyan-accent) !important;
    box-shadow: 0 0 10px rgba(0, 229, 255, 0.5) !important;
}
div[data-baseweb="slider"] div[data-testid="stSliderTickBar"] {
    background: #142740 !important;
}

/* Dataframe and Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #1C3352 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: #08101E !important;
}
[data-testid="stDataFrame"] [role="grid"] {
    background: #08101E !important;
}

/* Custom Dark Table */
.dark-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: #08101E;
    color: #E2EEFD;
    border: 1px solid #1C3352;
    border-radius: 12px;
    overflow: hidden;
    margin: 12px 0;
    font-size: 0.86rem;
}
.dark-table th {
    background: #0E1B2E;
    color: var(--cyan-accent);
    padding: 11px 15px;
    border-bottom: 1px solid #1C3352;
    text-align: left;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.74rem;
    letter-spacing: 0.06em;
}
.dark-table td {
    background: #08101E;
    color: #E2EEFD;
    padding: 9px 15px;
    border-bottom: 1px solid rgba(28, 51, 82, 0.5);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
.dark-table tr:last-child td {
    border-bottom: none;
}
.dark-table tr:hover td {
    background: #0E1D33;
}

/* Telemetry Card Containers */
.telemetry-card {
    background: #0A1322;
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.telemetry-card-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--cyan-accent);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Alert Boxes Override */
[data-testid="stAlert"] {
    background: #0A1729 !important;
    border: 1px solid #1B3556 !important;
    border-left: 4px solid var(--cyan-accent) !important;
    border-radius: 10px !important;
    color: #D6E8FC !important;
}

/* Footer */
.expo-footer {
    text-align: center;
    padding: 24px 0 10px 0;
    color: var(--text-dim);
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    border-top: 1px solid rgba(0, 229, 255, 0.1);
    margin-top: 36px;
}
.expo-footer b { color: #A0C2E2; }
</style>
''', unsafe_allow_html=True)

# ==============================================================================
# DATA ENGINE & SIGNAL PROCESSING FUNCTIONS (PRESERVED 100%)
# ==============================================================================
@st.cache_data(show_spinner=False)
def read_signal(path, key):
    obj = loadmat(str(path))
    if key not in obj:
        raise KeyError(f'MATLAB variable {key} missing in {path.name}')
    x = np.asarray(obj[key], dtype=float).reshape(-1)
    if len(x) < 100 or not np.all(np.isfinite(x)):
        raise ValueError('Invalid or non-finite vibration data')
    return x

def get_load(load):
    out = {}
    errors = []
    for name, (filename, key) in FILES[load].items():
        path = DATA / filename
        if not path.is_file():
            errors.append(f'{load}: missing {filename}')
            continue
        try:
            out[name] = read_signal(path, key)
        except Exception as e:
            errors.append(f'{load}: {filename}: {e}')
    return out, errors

def features(x):
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    return {
        'RMS': rms,
        'Kurtosis': float(kurtosis(x, fisher=True, bias=True)),
        'Crest Factor': peak / rms if rms else 0.0,
        'Peak-to-Peak': float(np.ptp(x)),
        'Mean': float(np.mean(x)),
        'Std': float(np.std(x)),
        'Variance': float(np.var(x)),
        'Peak': peak
    }

def feature_frame(load, signals):
    return pd.DataFrame([{'Load': load, 'Condition': n, **features(signals[n])} for n in signals])

def fft(x, limit=3000):
    n = len(x)
    freq = np.fft.rfftfreq(n, 1 / FS)
    mag = 2 * np.abs(np.fft.rfft((x - np.mean(x)) * np.hanning(n))) / max(np.sum(np.hanning(n)), 1)
    mask = freq <= limit
    return freq[mask], mag[mask]

def peak_frame(f, a, n=10):
    ids, _ = find_peaks(a)
    ids = ids[f[ids] > 0]
    if len(ids) == 0:
        return pd.DataFrame(columns=['Rank', 'Frequency (Hz)', 'Amplitude'])
    ids = ids[np.argsort(a[ids])[-n:]][::-1]
    return pd.DataFrame({
        'Rank': np.arange(1, len(ids) + 1),
        'Frequency (Hz)': np.round(f[ids], 3),
        'Amplitude': np.round(a[ids], 6)
    })

def envelope(x, limit):
    sos = butter(4, [max(100, FS * 0.08), min(5000, FS * 0.42)], btype='bandpass', fs=FS, output='sos')
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    return fft(env, limit)

def style_ax(ax, title=None, xlabel=None, ylabel=None):
    """Styles Matplotlib axes with a high-end industrial dark navy & cyan aesthetic."""
    ax.set_facecolor('#0B1322')
    ax.figure.set_facecolor('#0B1322')
    ax.grid(True, linestyle='--', color='#19324E', alpha=0.5, linewidth=0.7)
    ax.tick_params(colors='#8EA5C0', labelsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color='#A2BDDC', fontsize=9.5, fontweight='600')
    else:
        ax.xaxis.label.set_color('#A2BDDC')
        ax.xaxis.label.set_fontsize(9.5)
        ax.xaxis.label.set_fontweight('600')
    if ylabel:
        ax.set_ylabel(ylabel, color='#A2BDDC', fontsize=9.5, fontweight='600')
    else:
        ax.yaxis.label.set_color('#A2BDDC')
        ax.yaxis.label.set_fontsize(9.5)
        ax.yaxis.label.set_fontweight('600')
    if title:
        ax.set_title(title, color='#F0F6FF', fontsize=10.5, fontweight='700', pad=10)
    else:
        ax.title.set_color('#F0F6FF')
        ax.title.set_fontsize(10.5)
        ax.title.set_fontweight('700')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#1C3554')
    ax.spines['bottom'].set_color('#1C3554')

def download(df, label, filename):
    st.download_button(
        label,
        df.to_csv(index=False).encode('utf-8'),
        file_name=filename,
        mime='text/csv'
    )

# ==============================================================================
# MAIN DASHBOARD HUD HEADER
# ==============================================================================
st.markdown('''
<div class="hero-hud">
    <div class="hero-top-bar">
        <div class="hero-status-pill">
            <span class="hero-pulse-dot"></span> SYSTEM ONLINE // TELEMETRY ACQUISITION ACTIVE
        </div>
        <div class="hero-tag-group">
            <span class="hero-pill-badge">CWRU BENCHMARK</span>
            <span class="hero-pill-badge">12,000 Hz SAMPLING</span>
            <span class="hero-pill-badge accent">EXPO 2026 EDITION</span>
        </div>
    </div>
    <div class="hero-main-row">
        <div class="hero-title-group">
            <h1>⚙️ BearingIQ <span class="cyan-glow">Pro</span></h1>
            <p class="hero-subtitle">
                Rotary Machinery Vibration Analytics • Fast Fourier Transforms • Envelope Demodulation • Time-Frequency Spectrograms
            </p>
        </div>
        <div class="hero-spec-chips">
            <div class="spec-chip">
                <span class="lbl">DSP PIPELINE</span>
                <span class="val">Multi-Domain</span>
            </div>
            <div class="spec-chip">
                <span class="lbl">NYQUIST LIMIT</span>
                <span class="val">6,000 Hz</span>
            </div>
            <div class="spec-chip">
                <span class="lbl">COMPLIANCE</span>
                <span class="val">ISO 10816-3</span>
            </div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR NAVIGATION & DATASET SELECTION
# ==============================================================================
st.sidebar.markdown('''
<div class="sidebar-brand-card">
    <div class="sidebar-logo">⚡ BEARING-IQ <span>PRO</span></div>
    <div class="sidebar-sub">Vibration Diagnostics Console</div>
</div>
''', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section-title">📁 1. Data Ingestion</div>', unsafe_allow_html=True)
source = st.sidebar.radio(
    'Vibration Data Source',
    ['Built-in CWRU dataset', 'Upload your own data'],
    label_visibility='collapsed'
)

st.sidebar.markdown('<div class="sidebar-section-title">🔬 2. Diagnostic Module</div>', unsafe_allow_html=True)
modules = [
    'AI Bearing Intelligence & Diagnosis Center',
    'AI Health Command Center',
    'Overview & Summary',
    'Time-Domain Analysis',
    'FFT Spectrum',
    'Time-Frequency Spectrogram',
    'Dominant Peaks',
    'Feature Comparison',
    'Envelope Spectrum',
    'Experimental Fault Pattern Match',
    'Early Warning Indicators',
    'Load Comparison'
]
module = st.sidebar.selectbox('Select Analysis Workflow', modules, label_visibility='collapsed')

is_upload = (source == 'Upload your own data')
if is_upload and module == 'Load Comparison':
    st.sidebar.warning('⚠️ Load comparison requires built-in multi-load CWRU recordings. Please switch module.')
    st.stop()

if not is_upload and module != 'Load Comparison':
    st.sidebar.markdown('<div class="sidebar-section-title">⚡ 3. Motor Operating Load</div>', unsafe_allow_html=True)
    load = st.sidebar.selectbox('Drive Motor Load', ['0 HP', '1 HP', '2 HP', '3 HP'], label_visibility='collapsed')
else:
    load = 'Uploaded data' if is_upload else 'All loads'

# Upload handling
if is_upload:
    st.sidebar.markdown('<div class="sidebar-section-title">📤 4. Upload Signal File</div>', unsafe_allow_html=True)
    upload = st.file_uploader(
        'Upload vibration file (MAT, CSV, TXT)',
        type=['mat', 'csv', 'txt'],
        max_upload_size=50,
        help='Upload raw MATLAB .mat binary files or ASCII CSV/TXT vibration recordings (up to 50 MB).'
    )
    if upload is None:
        st.info('👉 Upload a MATLAB (.mat), CSV, or TXT vibration recording from the sidebar to begin analysis. Files are processed locally in-memory.')
        st.stop()
    try:
        raw = upload.getvalue()
        if not isinstance(raw, (bytes, bytearray)):
            raise ValueError('Expected raw binary bytes from uploaded file.')

        filename_lower = upload.name.lower()

        # Robust detection of MATLAB .mat binary format:
        # 1. Filename ends with .mat or .mat.txt
        # 2. Raw bytes start with standard MATLAB magic header b'MATLAB'
        # 3. Or scipy.io.loadmat parses the binary buffer without error
        is_mat = False
        if filename_lower.endswith('.mat') or filename_lower.endswith('.mat.txt') or raw.startswith(b'MATLAB'):
            is_mat = True
        else:
            try:
                loadmat(BytesIO(raw))
                is_mat = True
            except Exception:
                is_mat = False

        if is_mat:
            try:
                mat = loadmat(BytesIO(raw))
            except Exception as mat_err:
                raise ValueError(f'Corrupt or unreadable MATLAB .mat file: {mat_err}') from mat_err

            options = {}
            for k, v in mat.items():
                # Filter internal MATLAB metadata
                if k.startswith('__'):
                    continue
                try:
                    arr = np.asarray(v)
                    # Require numeric values
                    if not np.issubdtype(arr.dtype, np.number):
                        continue
                    # Check for 1D array or 2D column/row vector (e.g. Nx1, 1xN) with >= 100 samples
                    if arr.ndim <= 2 and (arr.ndim == 1 or 1 in arr.shape) and arr.size >= 100:
                        vec = arr.reshape(-1).astype(float)
                        if np.all(np.isfinite(vec)):
                            options[k] = vec
                except Exception:
                    continue

            if not options:
                available_vars = [k for k in mat.keys() if not k.startswith('__')]
                if available_vars:
                    raise ValueError(
                        f'No 1D numeric vibration signal vector with >= 100 samples found in {upload.name}. '
                        f'Available variables in MAT file: {available_vars}'
                    )
                else:
                    raise ValueError(f'No numeric variables found in the uploaded MAT file ({upload.name}).')

            column = st.selectbox(
                'Select MATLAB Signal Variable',
                list(options.keys()),
                help='Choose the vibration measurement channel extracted from the .mat file.'
            )
            x = options[column]

        else:
            # CSV / TXT format handling with multiple encoding fallbacks to avoid UTF-8 decode issues
            is_csv = filename_lower.endswith('.csv')
            tab = None
            for enc in [None, 'utf-8', 'latin-1', 'cp1252']:
                try:
                    if is_csv:
                        tab = pd.read_csv(BytesIO(raw), sep=None, engine='python', encoding=enc)
                    else:
                        try:
                            tab = pd.read_csv(BytesIO(raw), sep=None, engine='python', header=None, comment='#', encoding=enc)
                        except Exception:
                            tab = pd.read_csv(BytesIO(raw), sep=r'\s+', engine='python', header=None, comment='#', encoding=enc)
                    if tab is not None and not tab.empty:
                        break
                except Exception:
                    continue

            if tab is None or tab.empty:
                raise ValueError(f'Could not parse numeric tabular vibration data from {upload.name}.')

            valid = {}
            for c in tab.columns:
                vals = pd.to_numeric(tab[c], errors='coerce')
                if vals.notna().sum() >= 100:
                    valid[str(c)] = vals.dropna().to_numpy(dtype=float)

            if not valid:
                raise ValueError('No numeric column with at least 100 samples found in uploaded table.')

            column = st.selectbox(
                'Select Vibration Amplitude Column',
                list(valid.keys()),
                help='Choose the column containing numeric vibration amplitudes.'
            )
            x = valid[column]

        if len(x) < 100 or len(x) > 3_000_000 or not np.all(np.isfinite(x)):
            raise ValueError('Signal must contain between 100 and 3,000,000 finite numeric samples.')

    except Exception as exc:
        st.error(f'Could not read uploaded signal: {exc}')
        st.stop()

    FS = st.number_input(
        'Sampling Frequency (Hz)',
        min_value=1000,
        max_value=100000,
        value=12000,
        step=1000,
        help='Enter the exact measurement sampling frequency.'
    )
    label = st.text_input('Signal Designation Label', value=str(column) if 'column' in locals() else 'Uploaded Signal', max_chars=80)
    signals = {label: x}
    errors = []
    st.info('ℹ️ Exploratory signal processing mode. Custom uploaded recordings are analysed purely through deterministic signal features.')
else:
    st.sidebar.markdown('''
    <div class="sidebar-telemetry-box">
        <div class="sidebar-telemetry-title">BENCHMARK SPECIFICATIONS</div>
        <div class="sidebar-telemetry-row"><span>Transducer:</span><b>Accelerometer</b></div>
        <div class="sidebar-telemetry-row"><span>Sensor Point:</span><b>Drive End (DE)</b></div>
        <div class="sidebar-telemetry-row"><span>Bearing Type:</span><b>SKF 6205-2RS</b></div>
        <div class="sidebar-telemetry-row"><span>Sampling Rate:</span><b>12,000 Hz</b></div>
    </div>
    ''', unsafe_allow_html=True)

# ==============================================================================
# MODULE: LOAD COMPARISON (0 HP - 3 HP)
# ==============================================================================
if module == 'Load Comparison':
    frames = []
    problems = []
    for load_name in FILES:
        sig, err = get_load(load_name)
        problems += err
        if sig:
            frames.append(feature_frame(load_name, sig))
    if problems:
        with st.expander('Data Loading Notices'):
            for err in problems:
                st.warning(err)
    if not frames:
        st.error('No data available. Check the data directory.')
        st.stop()

    df = pd.concat(frames, ignore_index=True)

    st.markdown('''
    <div class="module-hud-banner">
        <div class="module-breadcrumb">BENCHMARK MULTI-LOAD SUITE // 0 HP – 3 HP</div>
        <h2>⚡ Cross-Load Feature Comparison</h2>
        <p class="module-desc">Comparative statistical analysis of vibration energy and impulsiveness across varying mechanical motor loads.</p>
    </div>
    ''', unsafe_allow_html=True)

    a, b, c = st.columns(3)
    a.metric('Motor Operating Loads', f"{df['Load'].nunique()} / 4", delta='0, 1, 2, 3 HP Tested')
    b.metric('Loaded Recordings', len(df), delta='Complete CWRU Cohort')
    c.metric('Acquisition Rate', f'{FS:,} Hz', delta='Nyquist: 6,000 Hz')

    st.markdown('#### 📊 Parameter Variance Across Operating Loads')
    metric = st.selectbox('Select Statistical Feature for Comparison', ['RMS', 'Kurtosis', 'Crest Factor', 'Peak-to-Peak', 'Std', 'Variance', 'Peak', 'Mean'])
    pivot = df.pivot(index='Condition', columns='Load', values=metric).reindex(NAMES)
    st.bar_chart(pivot)

    st.markdown('#### 📋 Full Statistical Summary Table (All Loads)')
    st.markdown(df.to_html(index=False, classes="dark-table", border=0), unsafe_allow_html=True)

    download(df, '⬇ Export Cross-Load Comparison (CSV)', 'bearing_comparison_all_loads.csv')
    st.caption('Note: Feature values are calculated over each full vibration recording. Deviations capture both motor torque/speed differences and structural defect severity.')
    st.stop()

# Load signals for individual load module
signals, errors = (signals, errors) if is_upload else get_load(load)
if errors:
    with st.expander('Data Loading Notices', expanded=not bool(signals)):
        for err in errors:
            st.warning(err)
if not signals:
    st.error(f'No usable signals available for {load}. Target directory: {DATA}')
    st.stop()

df = feature_frame(load, signals)
maxrow = df.loc[df['RMS'].idxmax()]

# GLOBAL KPI ROW
k1, k2, k3, k4 = st.columns(4)
k1.metric('Motor Load State', load, delta='Operating Profile')
k2.metric('Active Channels', f"{len(signals)} Condition{'s' if len(signals) > 1 else ''}", delta=f"Cohort: {len(FILES[load]) if not is_upload and load in FILES else 1}")
k3.metric('Peak RMS Signature', maxrow['Condition'], delta=f"{maxrow['RMS']:.4f} g RMS")
k4.metric('Sampling Bandwidth', f'{FS:,} Hz', delta=f'Nyquist: {FS // 2:,} Hz')

st.markdown('<hr style="border-color: rgba(0, 229, 255, 0.12); margin: 18px 0 24px 0;">', unsafe_allow_html=True)

# ==============================================================================
# MODULE: AI BEARING INTELLIGENCE & DIAGNOSIS CENTER
# ==============================================================================
if module == 'AI Bearing Intelligence & Diagnosis Center':
    st.markdown('''
    <div class="module-hud-banner">
        <div class="module-breadcrumb">AUTONOMOUS DIAGNOSTICS // MACHINE LEARNING INFERENCE ENGINE</div>
        <h2>🤖 AI Bearing Intelligence & Diagnosis Center</h2>
        <p class="module-desc">Autonomous multi-task machine learning system for bearing fault identification, motor load estimation, and out-of-distribution anomaly detection.</p>
    </div>
    ''', unsafe_allow_html=True)

    # Determine target signal
    if is_upload:
        active_name = label
        x_target = x
        source_name = upload.name
        var_name = column
    else:
        active_name = st.selectbox('Select Bearing Condition for AI Diagnosis', list(signals.keys()), key='ai_target_cond')
        x_target = signals[active_name]
        source_name = f'CWRU Benchmark ({load})'
        var_name = FILES[load][active_name][1] if (load in FILES and active_name in FILES[load]) else active_name

    # Run ML Inference Pipeline
    with st.spinner('Running AI Inference & Feature Extraction...'):
        models = get_ai_models()
        diag = mle.predict_bearing_health(x_target, fs=FS, models=models)

    # --------------------------------------------------------------------------
    # A. Uploaded Signal Information
    # --------------------------------------------------------------------------
    st.markdown('#### 📡 Signal Telemetry & Acquisition Parameters')
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric('Data Source', source_name, delta='Active Input')
    i2.metric('Vibration Channel', var_name, delta='1D Float Vector')
    i3.metric('Sampling Rate', f'{FS:,} Hz', delta=f'Nyquist: {FS // 2:,} Hz')
    i4.metric('Sample Length', f'{len(x_target):,} pts', delta='Continuous Series')
    i5.metric('Time Duration', f'{len(x_target) / FS:.2f} s', delta='Acquisition')

    st.markdown('<hr style="border-color: rgba(0, 229, 255, 0.12); margin: 16px 0 20px 0;">', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # B & C. Bearing Condition & Operating Load Prediction
    # --------------------------------------------------------------------------
    col_fault, col_load = st.columns([1.25, 1])

    with col_fault:
        st.markdown('#### 🎯 Bearing Condition Classification')
        
        # Health status badge
        if diag['is_healthy']:
            st.markdown('''
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">PRIMARY STATUS</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                    <span style="color: #10B981;">🟢</span> Normal Baseline (Healthy)
                </div>
                <div style="font-size: 0.8rem; color: #94E7C7; margin-top: 4px;">No raceway impact pulses or localized spalling harmonics detected.</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            color_badge = '#F59E0B' if diag['predicted_fault'] == 'Inner Race Fault' else ('#06B6D4' if diag['predicted_fault'] == 'Ball Fault' else '#EF4444')
            st.markdown(f'''
            <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid {color_badge}; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="font-size: 0.72rem; color: {color_badge}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">ANOMALY ALERT // FAULT DETECTED</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                    <span style="color: {color_badge};">🔴</span> {diag['predicted_fault']}
                </div>
                <div style="font-size: 0.8rem; color: #FCA5A5; margin-top: 4px;">Localized micro-impact vibrations detected in high-frequency envelope spectrum.</div>
            </div>
            ''', unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        fc1.metric('Predicted Class', diag['predicted_fault'], delta=f"{diag['fault_confidence'] * 100:.1f}% Confidence")
        fc2.metric('Classification Status', diag['diagnostic_confidence'], delta='In-Distribution' if not diag['is_ood'] else 'Out-of-Distribution Warning')

        # Horizontal probability bar chart
        st.markdown('##### Class-wise Probability Distribution')
        fig_fault, ax_fault = plt.subplots(figsize=(6.5, 2.6))
        f_classes = list(diag['fault_probabilities'].keys())
        f_probs = [diag['fault_probabilities'][c] * 100 for c in f_classes]
        f_colors = [COLORS.get(c, '#00B4D8') for c in f_classes]
        
        bars = ax_fault.barh(f_classes, f_probs, color=f_colors, height=0.55, edgecolor=(1.0, 1.0, 1.0, 0.2), linewidth=0.8)
        for bar in bars:
            w = bar.get_width()
            ax_fault.text(w + 1.2, bar.get_y() + bar.get_height() / 2., f'{w:.1f}%', va='center', ha='left', color='#E0EEFD', fontsize=8.5, fontweight='700')
        ax_fault.set_xlim(0, 115)
        style_ax(ax_fault, title='Fault Class Posterior Probabilities (%)', xlabel='Confidence Probability (%)')
        fig_fault.tight_layout()
        st.pyplot(fig_fault)
        plt.close(fig_fault)

    with col_load:
        st.markdown('#### ⚡ Motor Operating Load Detection')
        
        if diag['load_status'] == 'Confident':
            st.markdown(f'''
            <div style="background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.35); border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="font-size: 0.72rem; color: #00E5FF; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">ESTIMATED MOTOR LOAD</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                    <span style="color: #00E5FF;">⚡</span> {diag['predicted_load']}
                </div>
                <div style="font-size: 0.8rem; color: #94E7F7; margin-top: 4px;">Independently estimated from shaft slip harmonics and spectral energy.</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid #F59E0B; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="font-size: 0.72rem; color: #F59E0B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">OPERATING LOAD STATE</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                    <span style="color: #F59E0B;">⚠️</span> Unknown / Uncertain Load
                </div>
                <div style="font-size: 0.8rem; color: #FCD34D; margin-top: 4px;">Confidence < 45% or OOD; not forcing an HP classification.</div>
            </div>
            ''', unsafe_allow_html=True)

        lc1, lc2 = st.columns(2)
        lc1.metric('Operating Load', diag['predicted_load'], delta=f"{diag['load_confidence'] * 100:.1f}% Confidence" if diag['load_status'] == 'Confident' else 'Uncertain (<45%)')
        lc2.metric('Load Model Status', diag['load_status'], delta='Independent ML')

        # Load probability horizontal bar chart
        st.markdown('##### Operating Load Probability Distribution')
        fig_load, ax_load = plt.subplots(figsize=(6, 2.6))
        l_classes = list(diag['load_probabilities'].keys())
        l_probs = [diag['load_probabilities'][c] * 100 for c in l_classes]
        load_palette = ['#00B4D8', '#0096C7', '#0077B6', '#023E8A']
        
        bars_l = ax_load.barh(l_classes, l_probs, color=load_palette, height=0.55, edgecolor=(1.0, 1.0, 1.0, 0.2), linewidth=0.8)
        for bar in bars_l:
            w = bar.get_width()
            ax_load.text(w + 1.2, bar.get_y() + bar.get_height() / 2., f'{w:.1f}%', va='center', ha='left', color='#E0EEFD', fontsize=8.5, fontweight='700')
        ax_load.set_xlim(0, 115)
        style_ax(ax_load, title='Load Estimation Probabilities (%)', xlabel='Probability (%)')
        fig_load.tight_layout()
        st.pyplot(fig_load)
        plt.close(fig_load)

    st.markdown('<hr style="border-color: rgba(0, 229, 255, 0.12); margin: 20px 0 24px 0;">', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # D. Multiple Fault Analysis (Experimental / Not Yet Validated)
    # --------------------------------------------------------------------------
    st.markdown('#### 🔬 Multiple Fault Co-Occurrence Analysis')
    st.info('⚠️ **Research / Experimental Notice:** The authentic Case Western Reserve University (CWRU) benchmark dataset only contains single-defect bearing recordings (single inner race, ball, or outer race fault). No simultaneous multi-defect ground truth recordings exist in the benchmark. The indicators below examine concurrent envelope harmonic peaks but are **Experimental / Not Yet Validated**.')

    mf1, mf2, mf3 = st.columns(3)
    mf1.metric('Harmonic Overlap Risk', diag['multi_fault_risk'], delta=diag['multi_fault_indication'])
    mf2.metric('Active Spectral Indicators', f"{len(diag['active_fault_indicators'])} Detected" if diag['active_fault_indicators'] else 'None (Nominal)', delta='BPFO / BPFI / BSF Bands')
    mf3.metric('Multi-Label Classification', 'Non-Validated (Demo)', delta='Requires Multi-Defect Data')

    if diag['active_fault_indicators']:
        st.caption('Detected Characteristic Spectral Signatures: **' + ' • '.join(diag['active_fault_indicators']) + '**')

    st.markdown('<hr style="border-color: rgba(0, 229, 255, 0.12); margin: 20px 0 24px 0;">', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # E. Signal Health & Feature Extraction Analysis
    # --------------------------------------------------------------------------
    st.markdown('#### 📊 Signal Health Metrics & Feature Extraction Engine')
    f_kpi1, f_kpi2, f_kpi3, f_kpi4 = st.columns(4)
    feats = diag['features']
    f_kpi1.metric('Vibration RMS Energy', f"{feats['rms']:.4f} g", delta='ISO 10816 Zone A/B' if feats['rms'] < 0.15 else 'Elevated Energy')
    f_kpi2.metric('Signal Kurtosis', f"{feats['kurtosis']:.2f}", delta='Gaussian ~ 0' if abs(feats['kurtosis']) < 1.0 else 'Sharp Impacts')
    f_kpi3.metric('Crest Factor', f"{feats['crest_factor']:.2f}", delta='Peak / RMS Ratio')
    f_kpi4.metric('Dominant Spectral Peak', f"{feats['dominant_freq']:.1f} Hz", delta='Highest Spectral Power')

    with st.expander('🔍 View All 31 Extracted Physical Signal Features'):
        f_tab1, f_tab2, f_tab3 = st.tabs(['Time-Domain (12)', 'Frequency-Domain (10)', 'Envelope-Domain (9)'])
        with f_tab1:
            t_data = [
                {'Feature': 'RMS (Root Mean Square)', 'Value': f"{feats['rms']:.5f}", 'Engineering Significance': 'Total vibration energy across signal bandwidth'},
                {'Feature': 'Mean', 'Value': f"{feats['mean']:.5f}", 'Engineering Significance': 'DC offset / baseline calibration'},
                {'Feature': 'Standard Deviation', 'Value': f"{feats['std']:.5f}", 'Engineering Significance': 'Dynamic AC vibration fluctuation'},
                {'Feature': 'Variance', 'Value': f"{feats['var']:.5f}", 'Engineering Significance': 'Total signal power variation'},
                {'Feature': 'Skewness', 'Value': f"{feats['skewness']:.5f}", 'Engineering Significance': 'Asymmetry of acceleration distribution'},
                {'Feature': 'Kurtosis', 'Value': f"{feats['kurtosis']:.5f}", 'Engineering Significance': 'Impulsiveness / sharpness of transient impacts (>3 indicates spalling)'},
                {'Feature': 'Peak Amplitude', 'Value': f"{feats['peak']:.5f}", 'Engineering Significance': 'Maximum absolute acceleration peak'},
                {'Feature': 'Peak-to-Peak (PTP)', 'Value': f"{feats['ptp']:.5f}", 'Engineering Significance': 'Total waveform excursion (max - min)'},
                {'Feature': 'Crest Factor', 'Value': f"{feats['crest_factor']:.5f}", 'Engineering Significance': 'Ratio of peak to RMS; sensitive to early-stage spalling'},
                {'Feature': 'Shape Factor', 'Value': f"{feats['shape_factor']:.5f}", 'Engineering Significance': 'RMS / Mean Absolute ratio'},
                {'Feature': 'Impulse Factor', 'Value': f"{feats['impulse_factor']:.5f}", 'Engineering Significance': 'Peak / Mean Absolute ratio'},
                {'Feature': 'Margin Factor', 'Value': f"{feats['margin_factor']:.5f}", 'Engineering Significance': 'Peak to square-root mean amplitude ratio'}
            ]
            st.dataframe(pd.DataFrame(t_data), hide_index=True, use_container_width=True)

        with f_tab2:
            fq_data = [
                {'Feature': 'Dominant Frequency', 'Value': f"{feats['dominant_freq']:.2f} Hz", 'Engineering Significance': 'Frequency component with highest amplitude'},
                {'Feature': 'Spectral Centroid', 'Value': f"{feats['spectral_centroid']:.2f} Hz", 'Engineering Significance': 'Center of mass of frequency spectrum'},
                {'Feature': 'Spectral Spread', 'Value': f"{feats['spectral_spread']:.2f} Hz", 'Engineering Significance': 'Standard deviation of spectral power distribution'},
                {'Feature': 'Spectral Skewness', 'Value': f"{feats['spectral_skewness']:.4f}", 'Engineering Significance': 'Spectral asymmetry'},
                {'Feature': 'Spectral Kurtosis', 'Value': f"{feats['spectral_kurtosis']:.4f}", 'Engineering Significance': 'Frequency-domain peakiness (crucial for bearing defects)'},
                {'Feature': 'Spectral Energy', 'Value': f"{feats['spectral_energy']:.4f}", 'Engineering Significance': 'Integrated FFT power'},
                {'Feature': 'Band 0–100 Hz Energy Ratio', 'Value': f"{feats['band_0_100_ratio']:.4f}", 'Engineering Significance': 'Rotational speed & sub-synchronous slip harmonics'},
                {'Feature': 'Band 100–500 Hz Energy Ratio', 'Value': f"{feats['band_100_500_ratio']:.4f}", 'Engineering Significance': 'Fundamental defect frequencies (BPFO, BPFI, BSF)'},
                {'Feature': 'Band 500–2000 Hz Energy Ratio', 'Value': f"{feats['band_500_2000_ratio']:.4f}", 'Engineering Significance': 'Intermediate defect harmonics'},
                {'Feature': 'Band 2000–5000 Hz Energy Ratio', 'Value': f"{feats['band_2000_5000_ratio']:.4f}", 'Engineering Significance': 'High-frequency structural carrier resonances'}
            ]
            st.dataframe(pd.DataFrame(fq_data), hide_index=True, use_container_width=True)

        with f_tab3:
            env_data = [
                {'Feature': 'Envelope RMS', 'Value': f"{feats['env_rms']:.5f}", 'Engineering Significance': 'Demodulated high-frequency resonance RMS'},
                {'Feature': 'Envelope Peak', 'Value': f"{feats['env_peak']:.5f}", 'Engineering Significance': 'Maximum demodulated envelope impulse'},
                {'Feature': 'Envelope Crest Factor', 'Value': f"{feats['env_crest']:.4f}", 'Engineering Significance': 'Envelope peak to RMS ratio'},
                {'Feature': 'Envelope Kurtosis', 'Value': f"{feats['env_kurtosis']:.4f}", 'Engineering Significance': 'Impulsiveness of demodulated signal'},
                {'Feature': 'Envelope Dominant Frequency', 'Value': f"{feats['env_dominant_freq']:.2f} Hz", 'Engineering Significance': 'Modulation defect repetition rate'},
                {'Feature': 'Envelope Spectral Energy', 'Value': f"{feats['env_spectral_energy']:.4f}", 'Engineering Significance': 'Total demodulated spectral power'},
                {'Feature': 'BPFO Band Energy (95–118 Hz)', 'Value': f"{feats['env_bpfo_band_energy']:.5f}", 'Engineering Significance': 'Outer race pass harmonic energy (~105 Hz)'},
                {'Feature': 'BPFI Band Energy (148–172 Hz)', 'Value': f"{feats['env_bpfi_band_energy']:.5f}", 'Engineering Significance': 'Inner race pass harmonic energy (~160 Hz)'},
                {'Feature': 'BSF Band Energy (128–148 Hz)', 'Value': f"{feats['env_bsf_band_energy']:.5f}", 'Engineering Significance': 'Ball spin harmonic energy (~140 Hz)'}
            ]
            st.dataframe(pd.DataFrame(env_data), hide_index=True, use_container_width=True)

    st.markdown('<hr style="border-color: rgba(0, 229, 255, 0.12); margin: 20px 0 24px 0;">', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # F. Final AI Diagnosis Report & Interpretability
    # --------------------------------------------------------------------------
    st.markdown('#### 📑 Final AI Diagnostic Assessment & Engineering Narrative')
    
    # Out of Distribution Warning
    if diag['is_ood']:
        st.markdown(f'''
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; border-left: 5px solid #EF4444; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="font-size: 0.76rem; color: #FCA5A5; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">OUT-OF-DISTRIBUTION (OOD) DETECTED</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">Anomaly Score: {diag['ood_score']:.3f} &lt; 0.0</div>
            <div style="font-size: 0.84rem; color: #FCA5A5; line-height: 1.5;">
                This signal's multi-domain feature vector deviates from the authentic CWRU 12 kHz training distribution. 
                Possible causes include a non-standard sampling rate, different sensor accelerometer mounting position, different machinery geometry, or non-stationary noise.
                <b>Predictions must be verified with physical machine inspection and cannot be certified autonomously.</b>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="telemetry-card" style="border-left: 4px solid var(--cyan-accent); padding: 18px 22px;">
        <div class="telemetry-card-title">⚙️ Machine Learning Interpretability & Decision Rationale</div>
        <p style="font-size: 0.88rem; color: #C6E0FD; line-height: 1.6; margin: 6px 0 0 0;">
            {diag['technical_narrative']}
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # Download Full AI Diagnosis Report
    rep_data = {
        'Source File': source_name,
        'Signal Channel': var_name,
        'Sampling Rate (Hz)': FS,
        'Predicted Fault Condition': diag['predicted_fault'],
        'Fault Confidence (%)': round(diag['fault_confidence'] * 100, 2),
        'Health Status': diag['health_status'],
        'Estimated Motor Load': diag['predicted_load'],
        'Load Confidence (%)': round(diag['load_confidence'] * 100, 2),
        'Diagnostic Confidence': diag['diagnostic_confidence'],
        'OOD Anomaly Score': round(diag['ood_score'], 4),
        'Out of Distribution Flag': diag['is_ood'],
        'Multi-Fault Assessment': diag['multi_fault_indication'],
        **{f'Feature: {k}': round(v, 6) for k, v in feats.items()}
    }
    rep_df = pd.DataFrame([rep_data])
    download(rep_df, '⬇ Export Full AI Diagnosis Report (CSV)', f'ai_diagnosis_report_{source_name.replace(" ", "_")}.csv')

# ==============================================================================
# MODULE: AI HEALTH COMMAND CENTER
# ==============================================================================
elif module == 'AI Health Command Center':
    st.markdown('''
    <div class="module-hud-banner">
        <div class="module-breadcrumb">EXECUTIVE TELEMETRY // REAL-TIME CONDITION MONITORING</div>
        <h2>🧠 Machine Health Command Center</h2>
        <p class="module-desc">Real-time vibration feature profiling, energy distribution breakdown, and baseline status overview.</p>
    </div>
    ''', unsafe_allow_html=True)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('#### 📈 Vibration Feature Profile')
        feature_choice = st.selectbox(
            'Feature to Compare',
            ['RMS', 'Kurtosis', 'Crest Factor', 'Peak-to-Peak', 'Std', 'Variance', 'Peak', 'Mean'],
            key='command_feature'
        )
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        names_list = df['Condition'].tolist()
        vals_list = df[feature_choice].to_numpy()
        bars = ax.bar(
            names_list,
            vals_list,
            color=[COLORS.get(v, '#00B4D8') for v in names_list],
            edgecolor=(1.0, 1.0, 1.0, 0.2),
            linewidth=0.8,
            width=0.55
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                h,
                f'{h:.3g}',
                ha='center',
                va='bottom',
                color='#94B3D4',
                fontsize=8,
                fontweight='600'
            )
        ax.set_ylabel(feature_choice)
        ax.tick_params(axis='x', labelrotation=10)
        style_ax(ax, title=f'{feature_choice} Variance by Bearing State | {load}')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown('#### 📋 Condition Snapshot')
        st.dataframe(df[['Condition', 'RMS', 'Kurtosis', 'Crest Factor']].round(4), use_container_width=True, hide_index=True)

        st.markdown('''
        <div class="telemetry-card">
            <div class="telemetry-card-title">📖 Engineering Interpretation Guide</div>
            <p style="font-size:0.82rem; color:#A4C2E0; margin:0 0 6px 0;">
                <b>RMS (Root Mean Square):</b> Quantifies total vibration energy. Spikes during general mechanical looseness and unbalance.
            </p>
            <p style="font-size:0.82rem; color:#A4C2E0; margin:0 0 6px 0;">
                <b>Kurtosis:</b> Measures impulsiveness / peakiness of the signal. Values significantly &gt; 3.0 indicate sharp repetitive impacts (early race pitting).
            </p>
            <p style="font-size:0.82rem; color:#A4C2E0; margin:0;">
                <b>Crest Factor:</b> Ratio of peak amplitude to RMS. Highlights sudden localized impacts before overall energy increases.
            </p>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="telemetry-card" style="margin-top: 14px;">
        <div class="telemetry-card-title">🔄 Signal Processing Pipeline Architecture</div>
        <div style="font-size:0.84rem; color:#A0C4E8; font-family:'JetBrains Mono', monospace; line-height:1.6;">
            [Vibration Transducer] ➔ [12 kHz A/D Sampling] ➔ [Bandpass / Pre-Filter] ➔ [FFT & Envelope Demodulation] ➔ [Statistical Feature Engine] ➔ [Expo Dashboard]
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.warning('⚠️ Research & Educational Demonstration Suite: This dashboard demonstrates signal-processing methodologies and does not replace certified industrial vibration safety hardware or remaining useful life (RUL) prognostic systems.')

# ==============================================================================
# MODULE: OVERVIEW & SUMMARY
# ==============================================================================
elif module == 'Overview & Summary':
    st.markdown('''
    <div class="module-hud-banner">
        <div class="module-breadcrumb">EXECUTIVE SUMMARY // DATASET OVERVIEW</div>
        <h2>📋 System Feature Matrix</h2>
        <p class="module-desc">Full statistical vibration feature matrix calculated for the active operating condition cohort.</p>
    </div>
    ''', unsafe_allow_html=True)

    st.dataframe(df, use_container_width=True, hide_index=True)
    download(df, '⬇ Export Feature Summary (CSV)', f'feature_summary_{load.replace(" ", "")}.csv')

    st.markdown('''
    <div class="telemetry-card" style="margin-top: 18px;">
        <div class="telemetry-card-title">⚙️ Analytical Workflow</div>
        <p style="font-size:0.86rem; color:#9EBFDF; margin:0;">
            1. Raw signal loading from CWRU Drive-End acceleration records.<br>
            2. Statistical time-domain moments calculation (RMS, Variance, Kurtosis, Crest Factor).<br>
            3. Frequency decomposition via discrete Fast Fourier Transform with Hanning windowing.<br>
            4. High-frequency resonance demodulation via Hilbert envelope extraction.
        </p>
    </div>
    ''', unsafe_allow_html=True)

# ==============================================================================
# MODULE: EXPERIMENTAL FAULT PATTERN MATCH
# ==============================================================================
elif module == 'Experimental Fault Pattern Match':
    st.markdown('''
    <div class="module-hud-banner">
        <div class="module-breadcrumb">RESEARCH PROTOTYPE // STATISTICAL VECTOR SIMILARITY</div>
        <h2>🎯 Experimental Fault Pattern Match</h2>
        <p class="module-desc">Normalized multidimensional distance comparison against known CWRU reference feature vectors.</p>
    </div>
    ''', unsafe_allow_html=True)

    st.warning('⚠️ Research/Demo Tool Only: This pattern match demonstrates normalized distance metrics and is NOT an automated diagnostic decision or certified ML model.')

    if is_upload and FS != 12000:
        st.error('Reference CWRU signals are sampled at 12 kHz. For different sample rates, normalized distance matching is invalid; resampling and domain validation are required.')
        st.stop()

    if not is_upload and load in ('2 HP', '3 HP'):
        st.info('Reference comparison uses available recordings only. Missing conditions cannot be reliably matched.')

    references, problems = get_load('0 HP')
    refs1, problems1 = get_load('1 HP')
    if not references and not refs1:
        st.error('Reference CWRU recordings are unavailable in the data directory.')
        st.stop()

    feat_names = ['RMS', 'Kurtosis', 'Crest Factor', 'Peak-to-Peak']
    rows = []
    for ln, data in [('0 HP', references), ('1 HP', refs1)]:
        for condition, sig in data.items():
            rows.append({'Load': ln, 'Condition': condition, **features(sig)})
    reference_df = pd.DataFrame(rows)

    target_name = st.selectbox('Select Signal Vector for Distance Comparison', list(signals))
    target = features(signals[target_name])

    baseline = reference_df[feat_names].to_numpy(dtype=float)
    center = np.median(baseline, axis=0)
    scale = np.std(baseline, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    distances = np.sqrt(np.mean(((baseline - np.array([target[k] for k in feat_names])) / scale) ** 2, axis=1))

    reference_df['Relative feature distance'] = distances
    ranking = reference_df.sort_values('Relative feature distance').reset_index(drop=True)
    nearest = ranking.iloc[0]

    m1, m2 = st.columns([1, 1.5])
    m1.metric('Closest Reference Pattern', str(nearest['Condition']), delta=f"Load: {nearest['Load']}")
    m2.metric('Feature Distance', f"{nearest['Relative feature distance']:.4f}", delta='Smaller value = Closer signature match')

    st.markdown('#### 📊 Relative Distance Rankings')
    st.dataframe(ranking[['Load', 'Condition', 'Relative feature distance']].round(4), hide_index=True, use_container_width=True)

    download(ranking, '⬇ Export Reference Comparison (CSV)', 'experimental_pattern_match.csv')

# ==============================================================================
# MODULE: EARLY WARNING INDICATORS
# ==============================================================================
elif module == 'Early Warning Indicators':
    st.markdown('''
    <div class="module-hud-banner">
        <div class="module-breadcrumb">TRANSIENT TRENDS // TIME-WINDOW METRICS</div>
        <h2>⏱️ Early Warning Indicators • Within-Recording Trend</h2>
        <p class="module-desc">Segmented rolling window analysis to identify localized vibration energy jumps or impulsive spikes.</p>
    </div>
    ''', unsafe_allow_html=True)

    st.warning('⚠️ Anomaly indicators assess relative changes within this recording only; they do not predict time-to-failure or equipment remaining life.')

    name = st.selectbox('Select Signal for Rolling Trend Evaluation', list(signals))
    x = signals[name]
    nwin = min(max(256, int(FS * 0.25)), len(x) // 4)
    if nwin < 100:
        st.error('Insufficient samples for rolling window evaluation.')
        st.stop()

    count = min(len(x) // nwin, 1000)
    chunks = x[:count * nwin].reshape(count, nwin)
    rms = np.sqrt(np.mean(chunks ** 2, axis=1))
    crest = np.max(np.abs(chunks), axis=1) / np.maximum(rms, 1e-12)
    kur = kurtosis(chunks, axis=1, fisher=True, bias=True)
    times = (np.arange(count) + 0.5) * nwin / FS

    baseline_count = max(2, min(count // 4, 10))
    base = float(np.median(rms[:baseline_count]))
    ratio = rms / max(base, 1e-12)

    threshold = st.slider('Illustrative RMS Warning Threshold (× Initial Baseline)', 1.2, 4.0, 1.5, 0.1)
    warning_count = int(np.sum(ratio > threshold))

    w1, w2, w3 = st.columns(3)
    w1.metric('Initial Baseline RMS', f'{base:.5g}')
    w2.metric('Peak RMS / Baseline Ratio', f'{np.max(ratio):.2f}×')
    w3.metric('Windows Exceeding Threshold', f'{warning_count} / {count}', delta=f'Threshold: {threshold}×')

    if warning_count:
        st.warning(f'⚠️ Notice: {warning_count} time windows exceed {threshold}× of the initial reference baseline.')
    else:
        st.success('✅ Nominal: No time windows exceed the selected RMS threshold within this recording.')

    trend = pd.DataFrame({
        'Time (s)': times,
        'RMS': rms,
        'RMS / initial baseline': ratio,
        'Crest Factor': crest,
        'Kurtosis': kur
    })

    st.markdown('#### 📈 RMS Energy Ratio Progression Over Time')
    st.line_chart(trend.set_index('Time (s)')[['RMS / initial baseline']])

    st.markdown('#### 📈 Impulsive Indicators (Crest Factor & Kurtosis)')
    st.line_chart(trend.set_index('Time (s)')[['Crest Factor', 'Kurtosis']])

    st.dataframe(trend.round(5), hide_index=True, use_container_width=True)
    download(trend, '⬇ Export Windowed Trend Indicators (CSV)', 'vibration_early_warning_indicators.csv')

# ==============================================================================
# MODULE: FEATURE COMPARISON
# ==============================================================================
elif module == 'Feature Comparison':
    st.markdown('''
    <div class="module-hud-banner">
        <div class="module-breadcrumb">STATISTICAL COMPARISON // MULTI-METRIC VIEW</div>
        <h2>📊 Feature Comparison Across Conditions</h2>
        <p class="module-desc">Side-by-side engineering comparison of individual time-domain statistical metrics.</p>
    </div>
    ''', unsafe_allow_html=True)

    chosen = st.multiselect(
        'Select Vibration Features to Compare',
        ['RMS', 'Kurtosis', 'Crest Factor', 'Peak-to-Peak', 'Std', 'Variance', 'Peak', 'Mean'],
        default=['RMS', 'Kurtosis', 'Crest Factor']
    )
    if chosen:
        for metric in chosen:
            fig, ax = plt.subplots(figsize=(10, 2.8))
            names = df['Condition'].tolist()
            values = df[metric].to_numpy()
            bars = ax.bar(
                names,
                values,
                color=[COLORS.get(n, '#00B4D8') for n in names],
                edgecolor=(1.0, 1.0, 1.0, 0.2),
                linewidth=0.8,
                width=0.55
            )
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    h,
                    f'{h:.3g}',
                    ha='center',
                    va='bottom',
                    color='#94B3D4',
                    fontsize=8,
                    fontweight='600'
                )
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', labelrotation=10)
            style_ax(ax, title=f'{metric} Comparison | {load}')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.dataframe(df, use_container_width=True, hide_index=True)
    download(df, '⬇ Export Statistical Features (CSV)', f'features_{load.replace(" ", "")}.csv')

# ==============================================================================
# INDIVIDUAL SIGNAL MODULES (TIME, FFT, SPECTROGRAM, PEAKS, ENVELOPE)
# ==============================================================================
else:
    name = st.selectbox('Select Bearing Condition for Deep-Dive Analysis', [n for n in signals])
    x = signals[name]
    color = COLORS.get(name, '#00E5FF')

    # TIME-DOMAIN ANALYSIS
    if module == 'Time-Domain Analysis':
        st.markdown(f'''
        <div class="module-hud-banner">
            <div class="module-breadcrumb">TIME DOMAIN // RAW OSCILLOSCOPE WAVEFORM</div>
            <h2>📈 Time-Domain Analysis • {name}</h2>
            <p class="module-desc">Continuous raw acceleration time series captured at 12,000 samples per second.</p>
        </div>
        ''', unsafe_allow_html=True)

        maxdur = min(1.0, len(x) / FS)
        duration = st.slider('Display Window Duration (Seconds)', min_value=0.05, max_value=max(0.05, maxdur), value=min(0.25, maxdur), step=0.05)
        n = min(len(x), int(duration * FS))

        fig, ax = plt.subplots(figsize=(12, 4))
        t_arr = np.arange(n) / FS
        ax.plot(t_arr, x[:n], lw=1.1, color=color)
        ax.fill_between(t_arr, x[:n], alpha=0.12, color=color)
        style_ax(ax, title=f'Raw Vibration Acceleration | {name} | {load}', xlabel='Time (Seconds)', ylabel='Acceleration (g)')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Quick stats row for current signal
        f_sig = features(x)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric('Signal RMS', f"{f_sig['RMS']:.4f} g")
        s2.metric('Peak Amplitude', f"{f_sig['Peak']:.4f} g")
        s3.metric('Kurtosis', f"{f_sig['Kurtosis']:.3f}")
        s4.metric('Crest Factor', f"{f_sig['Crest Factor']:.3f}")

    # FFT SPECTRUM
    elif module == 'FFT Spectrum':
        st.markdown(f'''
        <div class="module-hud-banner">
            <div class="module-breadcrumb">FREQUENCY DOMAIN // FAST FOURIER TRANSFORM</div>
            <h2>⚡ FFT Spectrum Analysis • {name}</h2>
            <p class="module-desc">Spectral amplitude distribution revealing rotational harmonics and fault frequencies.</p>
        </div>
        ''', unsafe_allow_html=True)

        limit = st.slider('Maximum Spectral Frequency (Hz)', min(500, FS // 2), max(500, FS // 2), min(3000, FS // 2), 250)
        f, amp = fft(x, limit)

        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.plot(f, amp, color=color, lw=1.1)
        ax.fill_between(f, amp, alpha=0.15, color=color)
        style_ax(ax, title=f'FFT Amplitude Spectrum | {name} | {load}', xlabel='Frequency (Hz)', ylabel='Amplitude')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        peaks = peak_frame(f, amp)
        st.markdown('#### 🎯 Top 10 Identified Spectral Peaks')
        st.dataframe(peaks, use_container_width=True, hide_index=True)
        download(peaks, '⬇ Export FFT Spectral Peaks (CSV)', 'fft_peaks.csv')

    # TIME-FREQUENCY SPECTROGRAM
    elif module == 'Time-Frequency Spectrogram':
        st.markdown(f'''
        <div class="module-hud-banner">
            <div class="module-breadcrumb">TIME-FREQUENCY DOMAIN // STFT SPECTROGRAM</div>
            <h2>🌊 Time-Frequency Spectrogram • {name}</h2>
            <p class="module-desc">Short-Time Fourier Transform showing non-stationary vibration energy distribution over time and frequency.</p>
        </div>
        ''', unsafe_allow_html=True)

        f, t, p = spectrogram(x, fs=FS, nperseg=512, noverlap=448)
        sel = f <= min(3000, FS / 2)

        fig, ax = plt.subplots(figsize=(12, 4.6))
        m = ax.pcolormesh(t, f[sel], 10 * np.log10(p[sel] + 1e-15), cmap='magma', shading='auto')
        cbar = fig.colorbar(m, ax=ax, pad=0.02)
        cbar.set_label('Power Spectral Density (dB/Hz)', color='#A2BDDC', fontsize=9)
        cbar.ax.tick_params(colors='#8EA5C0', labelsize=8)
        cbar.outline.set_edgecolor('#1C3554')
        style_ax(ax, title=f'Spectrogram Energy Distribution | {name} | {load}', xlabel='Time (Seconds)', ylabel='Frequency (Hz)')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption('Tip: Pitting and spalling produce periodic wideband energy bursts visible as vertical stripes in the spectrogram.')

    # DOMINANT PEAKS
    elif module == 'Dominant Peaks':
        st.markdown(f'''
        <div class="module-hud-banner">
            <div class="module-breadcrumb">HARMONIC EXTRACTION // PEAK DETECTION</div>
            <h2>📍 Dominant FFT Peaks • {name}</h2>
            <p class="module-desc">Automatic peak detection identifying the highest-energy frequency components.</p>
        </div>
        ''', unsafe_allow_html=True)

        limit = st.slider('Maximum Frequency Bound (Hz)', min(500, FS // 2), max(500, FS // 2), min(3000, FS // 2), 250)
        f, amp = fft(x, limit)
        peaks = peak_frame(f, amp, 12)

        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.plot(f, amp, color=color, lw=1.1)
        ax.fill_between(f, amp, alpha=0.12, color=color)
        if not peaks.empty:
            ax.scatter(
                peaks['Frequency (Hz)'],
                peaks['Amplitude'],
                color='#00E5FF',
                edgecolors='#FFFFFF',
                s=40,
                zorder=5,
                linewidth=0.8,
                label='Detected Peak'
            )
            ax.legend(loc='upper right', facecolor='#0B1322', edgecolor='#1C3554', labelcolor='#D6E8FC', fontsize=8.5)
        style_ax(ax, title=f'Dominant Peak Spectrum | {name} | {load}', xlabel='Frequency (Hz)', ylabel='Amplitude')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown('#### 🎯 Top 12 Dominant Peaks Ranked by Amplitude')
        st.dataframe(peaks, use_container_width=True, hide_index=True)
        download(peaks, '⬇ Export Dominant Peaks (CSV)', 'dominant_peaks.csv')

    # ENVELOPE SPECTRUM
    elif module == 'Envelope Spectrum':
        st.markdown(f'''
        <div class="module-hud-banner">
            <div class="module-breadcrumb">DEMODULATION DSP // HILBERT TRANSFORM</div>
            <h2>🔍 Envelope Spectrum (High-Frequency Resonance) • {name}</h2>
            <p class="module-desc">Bandpass filtering (100–5000 Hz) followed by Hilbert transform envelope demodulation to extract impact fault frequencies.</p>
        </div>
        ''', unsafe_allow_html=True)

        limit = st.slider('Maximum Envelope Frequency (Hz)', 100, min(1000, FS // 2), min(300, FS // 2), 50)
        try:
            f, amp = envelope(x, limit)
            fig, ax = plt.subplots(figsize=(12, 4.2))
            ax.plot(f, amp, color=color, lw=1.1)
            ax.fill_between(f, amp, alpha=0.15, color=color)
            style_ax(ax, title=f'Demodulated Envelope Spectrum | {name} | {load}', xlabel='Envelope Frequency (Hz)', ylabel='Demodulated Amplitude')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            peaks = peak_frame(f, amp, 10)
            st.markdown('#### 🎯 Top 10 Envelope Harmonic Peaks')
            st.dataframe(peaks, use_container_width=True, hide_index=True)
            download(peaks, '⬇ Export Envelope Peaks (CSV)', 'envelope_peaks.csv')

            st.markdown('''
            <div class="telemetry-card" style="margin-top: 14px;">
                <div class="telemetry-card-title">💡 Why Envelope Demodulation Matters</div>
                <p style="font-size:0.84rem; color:#A4C2E0; margin:0;">
                    Localized bearing faults (inner race, outer race, ball) generate repetitive micro-impacts that modulate high-frequency structural resonances. 
                    Standard FFT often masks these low-frequency impact repetition rates under background shaft vibrations. Envelope demodulation strips away the carrier resonance, isolating the true fault repetition frequencies (BPFO, BPFI, BSF, FTF).
                </p>
            </div>
            ''', unsafe_allow_html=True)
        except Exception as e:
            st.error(f'Envelope processing error: {e}')

# ==============================================================================
# PRESENTATION FOOTER
# ==============================================================================
st.markdown('''
<div class="expo-footer">
    <b>BearingIQ Pro</b> • Industrial Rotating Machinery Condition Monitoring Suite • 
    Benchmark: Case Western Reserve University (CWRU) • Sampling: 12,000 Hz • College Project Expo Edition
</div>
''', unsafe_allow_html=True)
