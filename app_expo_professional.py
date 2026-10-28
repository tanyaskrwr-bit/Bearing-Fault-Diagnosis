"""Bearing Fault Diagnosis — Expo dashboard. Keep original app untouched.
Install: pip install streamlit numpy pandas matplotlib scipy
Run: python -m streamlit run app_expo_professional.py
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

FS = 12000
DATA = Path(__file__).resolve().parent / 'data'
NAMES = ['Normal', 'Inner Race Fault', 'Ball Fault', 'Outer Race Fault']
COLORS = {'Normal':'#2878B5','Inner Race Fault':'#E6862E','Ball Fault':'#269B62','Outer Race Fault':'#D64545'}
FILES = {
 '0 HP': {'Normal':('97.mat.txt','X097_DE_time'),'Inner Race Fault':('105.mat.txt','X105_DE_time'),'Ball Fault':('118.mat.txt','X118_DE_time'),'Outer Race Fault':('130.mat.txt','X130_DE_time')},
 '1 HP': {'Normal':('98.mat.txt','X098_DE_time'),'Inner Race Fault':('106.mat.txt','X106_DE_time'),'Ball Fault':('119.mat.txt','X119_DE_time'),'Outer Race Fault':('131.mat.txt','X131_DE_time')},
 '2 HP': {'Normal':('99.mat.txt','X099_DE_time'),'Inner Race Fault':('107.mat.txt','X107_DE_time')},
 '3 HP': {'Normal':('100.mat.txt','X100_DE_time'),'Inner Race Fault':('108.mat.txt','X108_DE_time')},
}
st.set_page_config(page_title='Bearing Fault Diagnosis | Engineering Dashboard',page_icon='⚙️',layout='wide')
st.markdown('''<style>
:root{color-scheme:dark}
.stApp{background:radial-gradient(ellipse at top right,#102743 0%,#07111f 48%,#050b14 100%);color:#e7f0ff}
.block-container{padding:1.5rem clamp(1rem,2.2vw,2.5rem) 2.5rem;max-width:none;width:100%}
[data-testid="stHeader"]{background:rgba(5,11,20,.78)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1627,#070e19);border-right:1px solid #1c3551}
[data-testid="stSidebar"] *{color:#dbeafe}
/* Keep every form control readable on the dark background */
div[data-baseweb="select"] > div{background:#0d1b2c !important;border:1px solid #365675 !important;color:#eaf4ff !important}
div[data-baseweb="select"] span,div[data-baseweb="select"] input{color:#eaf4ff !important;-webkit-text-fill-color:#eaf4ff !important}
div[data-baseweb="select"] svg{fill:#b8d8f2 !important}
ul[role="listbox"],div[role="listbox"],li[role="option"]{background:#102238 !important;color:#eaf4ff !important}
li[role="option"]:hover{background:#164267 !important}
[data-testid="stMetric"]{background:linear-gradient(145deg,#12243a,#0d1a2b);padding:17px 18px;border:1px solid #24425f;border-radius:15px;box-shadow:0 8px 24px #00000022}
[data-testid="stMetricLabel"]{color:#9bb4d0}
[data-testid="stMetricValue"]{color:#eaf6ff;font-size:clamp(1.25rem,2vw,2rem);white-space:normal;overflow-wrap:anywhere;line-height:1.2}
[data-testid="stMetricValue"] > div{white-space:normal;overflow-wrap:anywhere}
[data-testid="stMetricLabel"]{font-size:clamp(.85rem,1.1vw,1rem);white-space:normal}
[data-testid="stSidebar"]{min-width:220px}
[data-testid="stMetricDelta"]{color:#39d9ff}
.hero{background:linear-gradient(115deg,#0c1b2d 0%,#123456 58%,#07566a 100%);border:1px solid #2b6b91;color:#fff;padding:25px 28px;border-radius:18px;margin-bottom:20px;box-shadow:0 12px 32px #00000035;position:relative;overflow:hidden}
.hero:after{content:"";position:absolute;right:-55px;top:-95px;width:240px;height:240px;border:1px solid #53d9ff55;border-radius:50%;box-shadow:0 0 0 24px #53d9ff0b,0 0 0 48px #53d9ff08}
.hero h1{color:#fff;margin:0;font-size:2rem;letter-spacing:-.03em}.hero p{color:#b8d8f2;margin:7px 0 0;font-size:.98rem}
h1,h2,h3{letter-spacing:-.025em;color:#eaf4ff}
.stMarkdown,.stText,.stCaption{color:#c5d5e8}
hr{border-color:#20354d}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,div[data-baseweb="textarea"]>div{background:#0d1b2c;border-color:#29435f;color:#e7f0ff;border-radius:10px}
.stButton>button,.stDownloadButton>button{background:linear-gradient(100deg,#087ca8,#0b9ac1);color:#fff;border:1px solid #31c5ed;border-radius:10px;font-weight:600;transition:all .15s ease}
.stButton>button:hover,.stDownloadButton>button:hover{background:#0ca6cf;color:#fff;border-color:#7be7ff;box-shadow:0 0 18px #00bfff30}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid #294966;border-radius:12px;overflow:hidden;background:#0c1929}
[data-testid="stDataFrame"]{color-scheme:dark !important}
[data-testid="stDataFrame"] [role="grid"]{background:#0c1929 !important}
[data-testid="stDataFrame"] canvas{color-scheme:dark}
[data-testid="stTable"]{background:#0c1929 !important;color:#eaf4ff !important}
[data-testid="stTable"] table{width:100% !important;border-collapse:collapse !important;background:#0c1929 !important;color:#eaf4ff !important}
[data-testid="stTable"] th{background:#132940 !important;color:#cfe6ff !important;border:1px solid #294966 !important;padding:10px !important}
[data-testid="stTable"] td{background:#0c1929 !important;color:#eaf4ff !important;border:1px solid #294966 !important;padding:10px !important}
[data-testid="stDataFrame"] [data-testid="glideDataEditor"]{color-scheme:dark !important}
[data-testid="stTable"] table{background:#0c1929 !important;color:#e7f0ff !important}
[data-testid="stTable"] th{background:#132940 !important;color:#cfe6ff !important}
[data-testid="stTable"] td{background:#0c1929 !important;color:#e7f0ff !important}
[data-testid="stAlert"] p{color:inherit}
[data-testid="stAlert"]{border-radius:12px}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{background:#0e1d30;border-radius:9px 9px 0 0;padding:9px 15px}
.stTabs [aria-selected="true"]{border-bottom:2px solid #25c7f4;color:#61ddff}
.dark-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #0c1929;
    color: #e7f0ff;
}
.dark-table th {
    background-color: #132940;
    color: #cfe6ff;
    padding: 10px;
    border: 1px solid #29435c;
    text-align: left;
}
.dark-table td {
    background-color: #0c1929;
    color: #e7f0ff;
    padding: 9px;
    border: 1px solid #29435c;
}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_signal(path,key):
    obj=loadmat(str(path))
    if key not in obj: raise KeyError(f'MATLAB variable {key} missing in {path.name}')
    x=np.asarray(obj[key],dtype=float).reshape(-1)
    if len(x)<100 or not np.all(np.isfinite(x)): raise ValueError('Invalid or non-finite vibration data')
    return x

def get_load(load):
    out={}; errors=[]
    for name,(filename,key) in FILES[load].items():
        path=DATA/filename
        if not path.is_file(): errors.append(f'{load}: missing {filename}'); continue
        try: out[name]=read_signal(path,key)
        except Exception as e: errors.append(f'{load}: {filename}: {e}')
    return out,errors

def features(x):
    rms=float(np.sqrt(np.mean(x*x))); peak=float(np.max(np.abs(x)))
    return {'RMS':rms,'Kurtosis':float(kurtosis(x,fisher=True,bias=True)),
            'Crest Factor':peak/rms if rms else 0.,'Peak-to-Peak':float(np.ptp(x)),
            'Mean':float(np.mean(x)),'Std':float(np.std(x)),'Variance':float(np.var(x)),'Peak':peak}

def feature_frame(load, signals):
    return pd.DataFrame([{'Load':load,'Condition':n,**features(signals[n])} for n in signals])

def fft(x,limit=3000):
    n=len(x); freq=np.fft.rfftfreq(n,1/FS)
    mag=2*np.abs(np.fft.rfft((x-np.mean(x))*np.hanning(n)))/max(np.sum(np.hanning(n)),1)
    mask=freq<=limit
    return freq[mask],mag[mask]

def peak_frame(f,a,n=10):
    ids,_=find_peaks(a)
    ids=ids[f[ids]>0]
    if len(ids)==0: return pd.DataFrame(columns=['Rank','Frequency (Hz)','Amplitude'])
    ids=ids[np.argsort(a[ids])[-n:]][::-1]
    return pd.DataFrame({'Rank':np.arange(1,len(ids)+1),'Frequency (Hz)':np.round(f[ids],3),'Amplitude':np.round(a[ids],6)})

def envelope(x,limit):
    sos=butter(4,[max(100,FS*0.08),min(5000,FS*0.42)],btype='bandpass',fs=FS,output='sos')
    env=np.abs(hilbert(sosfiltfilt(sos,x)))
    return fft(env,limit)

def style_ax(ax):
    ax.set_facecolor('#0c1929')
    ax.figure.set_facecolor('#0c1929')
    ax.grid(alpha=.18,color='#7f9db9',linestyle='--')
    ax.tick_params(colors='#b9cde2')
    ax.xaxis.label.set_color('#c9d9eb'); ax.yaxis.label.set_color('#c9d9eb')
    ax.title.set_color('#eaf4ff')
    for spine in ax.spines.values(): spine.set_color('#29445e')

def download(df,label,filename):
    st.download_button(label,df.to_csv(index=False).encode('utf-8'),file_name=filename,mime='text/csv')

st.markdown('<div class="hero"><h1>⚙️ Bearing Fault Diagnosis</h1><p>INDUSTRIAL MACHINE HEALTH • CWRU VIBRATION ANALYTICS • ENGINEERING CONSOLE</p></div>',unsafe_allow_html=True)
st.sidebar.header('Dashboard Controls')
modules=['Overview & Summary','Time-Domain Analysis','FFT Spectrum','Time-Frequency Spectrogram','Dominant Peaks','Feature Comparison','Envelope Spectrum','Experimental Fault Pattern Match','Early Warning Indicators','Load Comparison']
source=st.sidebar.radio('Data source',['Built-in CWRU dataset','Upload your own data'])
module=st.sidebar.selectbox('Analysis module',modules)
is_upload=source=='Upload your own data'
if is_upload and module=='Load Comparison':
    st.sidebar.warning('Load comparison is for built-in CWRU data. Select another analysis.')
    st.stop()
load=st.sidebar.selectbox('Motor load',['0 HP','1 HP','2 HP','3 HP']) if not is_upload and module!='Load Comparison' else ('Uploaded data' if is_upload else 'All loads')
if is_upload:
    st.sidebar.caption('Upload CSV, TXT or MATLAB .mat (up to 50 MB).')
    upload=st.file_uploader('Upload a vibration signal',type=['csv','txt','mat'],max_upload_size=50)
    if upload is None:
        st.info('Upload a CSV, TXT, or MATLAB .mat vibration recording to begin. Files are processed in this session, not added to GitHub.')
        st.stop()
    try:
        raw=upload.getvalue()
        if upload.name.lower().endswith('.mat'):
            mat=loadmat(BytesIO(raw))
            options={k:np.asarray(v) for k,v in mat.items()
                     if not k.startswith('__') and np.issubdtype(np.asarray(v).dtype,np.number)
                     and np.asarray(v).size>=100 and np.asarray(v).ndim<=2
                     and 1 in np.asarray(v).shape}
            if not options: raise ValueError('No numeric vibration vector found in MAT file.')
            column=st.selectbox('MATLAB signal variable',list(options))
            x=np.asarray(options[column],dtype=float).reshape(-1)
        else:
            if upload.name.lower().endswith('.csv'):
                tab=pd.read_csv(BytesIO(raw),sep=None,engine='python')
            else:
                try: tab=pd.read_csv(BytesIO(raw),sep=None,engine='python',header=None,comment='#')
                except Exception: tab=pd.read_csv(BytesIO(raw),sep=r'\\s+',engine='python',header=None,comment='#')
            valid={}
            for c in tab.columns:
                vals=pd.to_numeric(tab[c],errors='coerce')
                if vals.notna().sum()>=100: valid[str(c)]=vals.dropna().to_numpy(dtype=float)
            if not valid: raise ValueError('No numeric column with at least 100 samples found.')
            column=st.selectbox('Vibration amplitude column',list(valid))
            x=valid[column]
        if len(x)<100 or len(x)>3_000_000 or not np.all(np.isfinite(x)):
            raise ValueError('Signal must contain 100–3,000,000 finite numeric samples.')
    except Exception as exc:
        st.error(f'Could not read uploaded signal: {exc}')
        st.stop()
    FS=st.number_input('Sampling frequency of uploaded signal (Hz)',min_value=1000,max_value=100000,value=12000,step=1000,
                       help='Enter the actual sample rate from the measurement equipment; do not assume 12 kHz.')
    label=st.text_input('Signal label',value='Uploaded signal',max_chars=80)
    signals={label:x}
    errors=[]
    st.info('Exploratory signal analysis only. Uploaded recordings are not automatically classified as Normal or Faulty.')
else:
    st.sidebar.caption('Built-in CWRU drive-end signals | 12,000 Hz')
    st.sidebar.info('2 HP / 3 HP use available Normal and Inner Race recordings only; missing data is not inferred.')

if module=='Load Comparison':
    frames=[]; problems=[]
    for load_name in FILES:
        sig,err=get_load(load_name);problems+=err
        if sig: frames.append(feature_frame(load_name,sig))
    if problems:
        with st.expander('Data loading notices'):
            for err in problems: st.warning(err)
    if not frames: st.error('No data available. Check the data folder.');st.stop()
    df=pd.concat(frames,ignore_index=True)
    a,b,c=st.columns(3)
    a.metric('Motor loads',f'{df.Load.nunique()} / 4');b.metric('Loaded recordings',len(df));c.metric('Sampling frequency',f'{FS:,} Hz')
    st.subheader('0–3 HP • Available Feature Comparison')
    metric=st.selectbox('Feature',['RMS','Kurtosis','Crest Factor','Peak-to-Peak','Std','Variance','Peak','Mean'])
    pivot=df.pivot(index='Condition',columns='Load',values=metric).reindex(NAMES)
    st.bar_chart(pivot)

    st.markdown(
        df.to_html(index=False, classes="dark-table", border=0),
        unsafe_allow_html=True
    )

    download(df, '⬇ Export both loads (CSV)', 'bearing_comparison_all_loads.csv')
    st.caption('Feature values use the full recording for each condition. Differences may reflect operating load as well as fault condition.')
    st.stop()

signals,errors=(signals,errors) if is_upload else get_load(load)
if errors:
    with st.expander('Data loading notices',expanded=not bool(signals)):
        for err in errors: st.warning(err)
if not signals:
    st.error(f'No usable signals for {load}. Required data folder: {DATA}')
    st.stop()
df=feature_frame(load,signals)
maxrow=df.loc[df['RMS'].idxmax()]
a,b,c,d=st.columns(4)
a.metric('Selected load',load);b.metric('Signals loaded',f'{len(signals)} / {1 if is_upload else 4}');c.metric('Highest RMS condition',maxrow['Condition']);d.metric('Sampling frequency',f'{FS:,} Hz')
st.caption('RMS, kurtosis and other features are calculated on each full vibration recording.')
st.divider()

if module=='Experimental Fault Pattern Match':
    st.subheader('Experimental Fault Pattern Match')
    st.warning('Research/demo tool only. This is NOT a validated ML classifier and must not be used to diagnose a machine or predict failure.')
    if is_upload and FS != 12000:
        st.error('Reference CWRU signals are sampled at 12 kHz. For different sample rates, this demo does not compare patterns; resampling and domain validation are required.')
        st.stop()
    if not is_upload and load in ('2 HP','3 HP'):
        st.info('Reference comparison uses available recordings only. Missing conditions cannot be reliably identified.')
    references, problems=get_load('0 HP')
    refs1, problems1=get_load('1 HP')
    if not references and not refs1:
        st.error('Reference CWRU recordings are unavailable in the data folder.')
        st.stop()
    # Each recording is one example, not an independent training/test population.
    feat_names=['RMS','Kurtosis','Crest Factor','Peak-to-Peak']
    rows=[]
    for ln, data in [('0 HP',references),('1 HP',refs1)]:
        for condition, sig in data.items():
            rows.append({'Load':ln,'Condition':condition,**features(sig)})
    reference_df=pd.DataFrame(rows)
    target_name=st.selectbox('Signal to compare',list(signals))
    target=features(signals[target_name])
    baseline=reference_df[feat_names].to_numpy(dtype=float)
    center=np.median(baseline,axis=0)
    scale=np.std(baseline,axis=0)
    scale=np.where(scale>1e-10,scale,1.)
    distances=np.sqrt(np.mean(((baseline-np.array([target[k] for k in feat_names]))/scale)**2,axis=1))
    reference_df['Relative feature distance']=distances
    ranking=reference_df.sort_values('Relative feature distance').reset_index(drop=True)
    nearest=ranking.iloc[0]
    st.metric('Closest reference pattern (NOT a diagnosis)',str(nearest['Condition']))
    st.caption(f"Closest example: {nearest['Load']}. Distance {nearest['Relative feature distance']:.3f} (smaller = more similar). No accuracy or confidence percentage is implied.")
    st.dataframe(ranking[['Load','Condition','Relative feature distance']].round(4),hide_index=True,use_container_width=True)
    download(ranking,'⬇ Export reference comparison','experimental_pattern_match.csv')
    st.info('A different machine, sensor, sampling rate, speed or load can invalidate this comparison. Train and independently validate a real model before enabling automated fault labels.')

elif module=='Early Warning Indicators':
    st.subheader('Early Warning Indicators • Within-recording trend')
    st.warning('Anomaly indicators only — not a prediction of when a fault will occur.')
    name=st.selectbox('Signal for trend analysis',list(signals))
    x=signals[name]
    nwin=min(max(256,int(FS*0.25)),len(x)//4)
    if nwin<100:
        st.error('Not enough samples for window analysis.')
        st.stop()
    count=min(len(x)//nwin,1000)
    chunks=x[:count*nwin].reshape(count,nwin)
    rms=np.sqrt(np.mean(chunks**2,axis=1))
    crest=np.max(np.abs(chunks),axis=1)/np.maximum(rms,1e-12)
    kur=kurtosis(chunks,axis=1,fisher=True,bias=True)
    times=(np.arange(count)+0.5)*nwin/FS
    baseline_count=max(2,min(count//4,10))
    base=float(np.median(rms[:baseline_count]))
    ratio=rms/max(base,1e-12)
    threshold=st.slider('Illustrative RMS increase threshold (× initial baseline)',1.2,4.0,1.5,0.1)
    warning_count=int(np.sum(ratio>threshold))
    a,b,c=st.columns(3)
    a.metric('Initial RMS baseline',f'{base:.5g}')
    b.metric('Maximum RMS / baseline',f'{np.max(ratio):.2f}×')
    c.metric('Windows above threshold',f'{warning_count} / {count}')
    if warning_count:
        st.warning('RMS increase detected relative to the start of THIS recording. Verify against a healthy-machine baseline and operating conditions.')
    else:
        st.success('No RMS threshold crossings within this recording. This does NOT guarantee healthy operation.')
    trend=pd.DataFrame({'Time (s)':times,'RMS':rms,'RMS / initial baseline':ratio,'Crest Factor':crest,'Kurtosis':kur})
    st.line_chart(trend.set_index('Time (s)')[['RMS / initial baseline']])
    st.line_chart(trend.set_index('Time (s)')[['Crest Factor','Kurtosis']])
    st.dataframe(trend.round(5),hide_index=True,use_container_width=True)
    download(trend,'⬇ Export trend indicators','vibration_early_warning_indicators.csv')
    st.caption('The initial windows are an arbitrary reference, not a verified healthy baseline. True predictive maintenance needs repeated timestamped measurements and failure histories.')

elif module=='Overview & Summary':
    st.subheader('System Overview')
    st.write('Explore vibration signatures with six signal-analysis modules. Built-in CWRU recordings have known labels; uploaded signals are unlabelled and are not automatically classified.')
    st.dataframe(df, use_container_width=True, hide_index=True)
    download(df,'⬇ Export feature summary (CSV)',f'feature_summary_{load.replace(" ","")}.csv')
    st.markdown('**Workflow:** Dataset / Upload → Signal loading → Signal processing → Feature extraction → Visual interpretation → Future validated diagnosis')
else:
    if module=='Feature Comparison':
        st.subheader('Feature Comparison')
        chosen=st.multiselect('Select features',['RMS','Kurtosis','Crest Factor','Peak-to-Peak','Std','Variance','Peak','Mean'],default=['RMS','Kurtosis','Crest Factor'])
        if chosen:
            # Separate scales avoid mixing incompatible engineering quantities.
            for metric in chosen:
                fig,ax=plt.subplots(figsize=(10,2.7))
                names=df['Condition'].tolist();values=df[metric].to_numpy()
                ax.bar(names,values,color=[COLORS.get(n,'#2878B5') for n in names]);ax.set_ylabel(metric);style_ax(ax)
                ax.tick_params(axis='x',labelrotation=12);fig.tight_layout();st.pyplot(fig);plt.close(fig)
        st.dataframe(df,use_container_width=True,hide_index=True)
        download(df,'⬇ Export features (CSV)',f'features_{load.replace(" ","")}.csv')
    else:
        name=st.selectbox('Bearing condition',[n for n in signals]);x=signals[name]
        color=COLORS.get(name,'#2878B5')
        if module=='Time-Domain Analysis':
            st.subheader('Time-Domain Analysis')
            maxdur=min(1.,len(x)/FS)
            duration=st.slider('Display duration (seconds)',min_value=.05,max_value=max(.05,maxdur),value=min(.25,maxdur),step=.05)
            n=min(len(x),int(duration*FS));fig,ax=plt.subplots(figsize=(12,4))
            ax.plot(np.arange(n)/FS,x[:n],lw=.75,color=color)
            ax.set(xlabel='Time (s)',ylabel='Vibration amplitude',title=f'{name} | {load}');style_ax(ax)
            fig.tight_layout();st.pyplot(fig);plt.close(fig)
        elif module=='FFT Spectrum':
            st.subheader('FFT Spectrum')
            limit=st.slider('Maximum frequency (Hz)',min(500,FS//2),max(500,FS//2),min(3000,FS//2),250)
            f,amp=fft(x,limit);fig,ax=plt.subplots(figsize=(12,4))
            ax.plot(f,amp,color=color,lw=.8);ax.set(xlabel='Frequency (Hz)',ylabel='Amplitude',title=f'FFT | {name} | {load}')
            style_ax(ax);fig.tight_layout();st.pyplot(fig);plt.close(fig)
            peaks=peak_frame(f,amp);st.dataframe(peaks,use_container_width=True,hide_index=True)
            download(peaks,'⬇ Export FFT peaks (CSV)','fft_peaks.csv')
        elif module=='Time-Frequency Spectrogram':
            st.subheader('Time-Frequency Spectrogram')
            f,t,p=spectrogram(x,fs=FS,nperseg=512,noverlap=448)
            sel=f<=min(3000,FS/2);fig,ax=plt.subplots(figsize=(12,4.5))
            m=ax.pcolormesh(t,f[sel],10*np.log10(p[sel]+1e-15),cmap='viridis',shading='auto')
            fig.colorbar(m,ax=ax,label='Power spectral density (dB/Hz)')
            ax.set(xlabel='Time (s)',ylabel='Frequency (Hz)',title=f'Spectrogram | {name} | {load}')
            fig.tight_layout();st.pyplot(fig);plt.close(fig)
        elif module=='Dominant Peaks':
            st.subheader('Dominant FFT Peaks')
            limit=st.slider('Maximum frequency (Hz)',min(500,FS//2),max(500,FS//2),min(3000,FS//2),250)
            f,amp=fft(x,limit);peaks=peak_frame(f,amp,12)
            fig,ax=plt.subplots(figsize=(12,4));ax.plot(f,amp,color=color,lw=.7)
            if not peaks.empty: ax.scatter(peaks['Frequency (Hz)'],peaks['Amplitude'],color='#102b49',s=20,zorder=4)
            ax.set(xlabel='Frequency (Hz)',ylabel='Amplitude',title=f'Dominant peaks | {name} | {load}')
            style_ax(ax);fig.tight_layout();st.pyplot(fig);plt.close(fig)
            st.dataframe(peaks,use_container_width=True,hide_index=True)
            download(peaks,'⬇ Export dominant peaks (CSV)','dominant_peaks.csv')
        elif module=='Envelope Spectrum':
            st.subheader('Envelope Spectrum')
            limit=st.slider('Maximum envelope frequency (Hz)',100,min(1000,FS//2),min(300,FS//2),50)
            try:
                f,amp=envelope(x,limit)
                fig,ax=plt.subplots(figsize=(12,4));ax.plot(f,amp,color=color,lw=.8)
                ax.set(xlabel='Envelope frequency (Hz)',ylabel='Envelope amplitude',title=f'Envelope spectrum | {name} | {load}')
                style_ax(ax);fig.tight_layout();st.pyplot(fig);plt.close(fig)
                peaks=peak_frame(f,amp,10)
                st.dataframe(peaks,use_container_width=True,hide_index=True,theme="dark")
                download(peaks,'⬇ Export envelope peaks (CSV)','envelope_peaks.csv')
                st.info('Envelope spectral peaks are diagnostic indicators, not automatic proof of a specific fault.')
            except Exception as e: st.error(f'Envelope processing error: {e}')

st.divider()
st.caption('CWRU / uploaded signals • Research demonstration only • No validated fault classifier or future-failure prediction.')
