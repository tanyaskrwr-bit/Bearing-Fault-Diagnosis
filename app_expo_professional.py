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
.stApp{background:#f4f7fb}.block-container{padding-top:1.5rem;max-width:1450px}
[data-testid="stSidebar"]{background:#eaf0f7}
[data-testid="stMetric"]{background:white;padding:15px;border:1px solid #dce5ef;border-radius:13px;box-shadow:0 3px 10px #1e3a5f0b}
.hero{background:linear-gradient(115deg,#102b49,#246078);color:white;padding:24px 28px;border-radius:16px;margin-bottom:20px}
.hero h1{color:white;margin:0;font-size:2rem}.hero p{color:#d9eaf4;margin:5px 0 0}
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
    ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)

def download(df,label,filename):
    st.download_button(label,df.to_csv(index=False).encode('utf-8'),file_name=filename,mime='text/csv')

st.markdown('<div class="hero"><h1>⚙️ Bearing Fault Diagnosis</h1><p>Industrial Machine Health Monitoring • CWRU Vibration Analytics • Engineering Dashboard</p></div>',unsafe_allow_html=True)
st.sidebar.header('Dashboard Controls')
modules=['Overview & Summary','Time-Domain Analysis','FFT Spectrum','Time-Frequency Spectrogram','Dominant Peaks','Feature Comparison','Envelope Spectrum','Load Comparison']
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
    st.sidebar.info('2 HP / 3 HP show available Normal and Inner Race data. Ball and Outer Race require additional files.')

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
    st.dataframe(df,use_container_width=True,hide_index=True)
    download(df,'⬇ Export both loads (CSV)','bearing_comparison_all_loads.csv')
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

if module=='Overview & Summary':
    st.subheader('System Overview')
    st.write('Explore vibration signatures with six signal-analysis modules. Built-in CWRU recordings have known labels; uploaded signals are unlabelled and are not automatically classified.')
    st.dataframe(df,use_container_width=True,hide_index=True)
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
                st.dataframe(peaks,use_container_width=True,hide_index=True)
                download(peaks,'⬇ Export envelope peaks (CSV)','envelope_peaks.csv')
                st.info('Envelope spectral peaks are diagnostic indicators, not automatic proof of a specific fault.')
            except Exception as e: st.error(f'Envelope processing error: {e}')

st.divider()
st.caption('Built-in CWRU dataset or user-uploaded vibration data • Exploratory signal analysis only; not validated automatic fault classification.')
