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
from scipy.signal import spectrogram, find_peaks, butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis

FS = 12000
DATA = Path(__file__).resolve().parent / 'data'
NAMES = ['Normal', 'Inner Race Fault', 'Ball Fault', 'Outer Race Fault']
COLORS = {'Normal':'#2878B5','Inner Race Fault':'#E6862E','Ball Fault':'#269B62','Outer Race Fault':'#D64545'}
FILES = {
 '0 HP': {'Normal':('97.mat.txt','X097_DE_time'),'Inner Race Fault':('105.mat.txt','X105_DE_time'),'Ball Fault':('118.mat.txt','X118_DE_time'),'Outer Race Fault':('130.mat.txt','X130_DE_time')},
 '1 HP': {'Normal':('98.mat.txt','X098_DE_time'),'Inner Race Fault':('106.mat.txt','X106_DE_time'),'Ball Fault':('119.mat.txt','X119_DE_time'),'Outer Race Fault':('131.mat.txt','X131_DE_time')},
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
    return pd.DataFrame([{'Load':load,'Condition':n,**features(signals[n])} for n in NAMES if n in signals])

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
    sos=butter(4,[1000,5000],btype='bandpass',fs=FS,output='sos')
    env=np.abs(hilbert(sosfiltfilt(sos,x)))
    return fft(env,limit)

def style_ax(ax):
    ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)

def download(df,label,filename):
    st.download_button(label,df.to_csv(index=False).encode('utf-8'),file_name=filename,mime='text/csv')

st.markdown('<div class="hero"><h1>⚙️ Bearing Fault Diagnosis</h1><p>Industrial Machine Health Monitoring • CWRU Vibration Analytics • Engineering Dashboard</p></div>',unsafe_allow_html=True)
st.sidebar.header('Dashboard Controls')
modules=['Overview & Summary','Time-Domain Analysis','FFT Spectrum','Time-Frequency Spectrogram','Dominant Peaks','Feature Comparison','Envelope Spectrum','0 HP vs 1 HP Comparison']
module=st.sidebar.selectbox('Analysis module',modules)
load=st.sidebar.radio('Motor load',['0 HP','1 HP'],horizontal=True) if module!='0 HP vs 1 HP Comparison' else 'Both'
st.sidebar.caption('Sampling frequency: 12,000 Hz | Drive-end signals')
st.sidebar.info('2 HP and 3 HP will be added after verifying all required dataset files.')

if module=='0 HP vs 1 HP Comparison':
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
    a.metric('Motor loads',f'{df.Load.nunique()} / 2');b.metric('Loaded recordings',len(df));c.metric('Sampling frequency','12 kHz')
    st.subheader('0 HP vs 1 HP • Feature Comparison')
    metric=st.selectbox('Feature',['RMS','Kurtosis','Crest Factor','Peak-to-Peak','Std','Variance','Peak','Mean'])
    pivot=df.pivot(index='Condition',columns='Load',values=metric).reindex(NAMES)
    st.bar_chart(pivot)
    st.dataframe(df,use_container_width=True,hide_index=True)
    download(df,'⬇ Export both loads (CSV)','bearing_comparison_0HP_1HP.csv')
    st.caption('Feature values use the full recording for each condition. Differences may reflect operating load as well as fault condition.')
    st.stop()

signals,errors=get_load(load)
if errors:
    with st.expander('Data loading notices',expanded=not bool(signals)):
        for err in errors: st.warning(err)
if not signals:
    st.error(f'No usable signals for {load}. Required data folder: {DATA}')
    st.stop()
df=feature_frame(load,signals)
maxrow=df.loc[df['RMS'].idxmax()]
a,b,c,d=st.columns(4)
a.metric('Selected load',load);b.metric('Signals loaded',f'{len(signals)} / 4');c.metric('Highest RMS condition',maxrow['Condition']);d.metric('Sampling frequency','12 kHz')
st.caption('RMS, kurtosis and other features are calculated on each full vibration recording.')
st.divider()

if module=='Overview & Summary':
    st.subheader('System Overview')
    st.write('Explore vibration signatures for four bearing conditions using six signal-analysis modules. The dashboard displays known labelled CWRU recordings; it does not yet automatically classify unseen vibration signals.')
    st.dataframe(df,use_container_width=True,hide_index=True)
    download(df,'⬇ Export feature summary (CSV)',f'feature_summary_{load.replace(" ","")}.csv')
    st.markdown('**Workflow:** Dataset → Signal loading → Signal processing → Feature extraction → Visual interpretation → Future ML-based diagnosis')
else:
    if module=='Feature Comparison':
        st.subheader('Feature Comparison')
        chosen=st.multiselect('Select features',['RMS','Kurtosis','Crest Factor','Peak-to-Peak','Std','Variance','Peak','Mean'],default=['RMS','Kurtosis','Crest Factor'])
        if chosen:
            # Separate scales avoid mixing incompatible engineering quantities.
            for metric in chosen:
                fig,ax=plt.subplots(figsize=(10,2.7))
                names=df['Condition'].tolist();values=df[metric].to_numpy()
                ax.bar(names,values,color=[COLORS[n] for n in names]);ax.set_ylabel(metric);style_ax(ax)
                ax.tick_params(axis='x',labelrotation=12);fig.tight_layout();st.pyplot(fig);plt.close(fig)
        st.dataframe(df,use_container_width=True,hide_index=True)
        download(df,'⬇ Export features (CSV)',f'features_{load.replace(" ","")}.csv')
    else:
        name=st.selectbox('Bearing condition',[n for n in NAMES if n in signals]);x=signals[name]
        color=COLORS[name]
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
            limit=st.slider('Maximum frequency (Hz)',500,6000,3000,250)
            f,amp=fft(x,limit);fig,ax=plt.subplots(figsize=(12,4))
            ax.plot(f,amp,color=color,lw=.8);ax.set(xlabel='Frequency (Hz)',ylabel='Amplitude',title=f'FFT | {name} | {load}')
            style_ax(ax);fig.tight_layout();st.pyplot(fig);plt.close(fig)
            peaks=peak_frame(f,amp);st.dataframe(peaks,use_container_width=True,hide_index=True)
            download(peaks,'⬇ Export FFT peaks (CSV)','fft_peaks.csv')
        elif module=='Time-Frequency Spectrogram':
            st.subheader('Time-Frequency Spectrogram')
            f,t,p=spectrogram(x,fs=FS,nperseg=512,noverlap=448)
            sel=f<=3000;fig,ax=plt.subplots(figsize=(12,4.5))
            m=ax.pcolormesh(t,f[sel],10*np.log10(p[sel]+1e-15),cmap='viridis',shading='auto')
            fig.colorbar(m,ax=ax,label='Power spectral density (dB/Hz)')
            ax.set(xlabel='Time (s)',ylabel='Frequency (Hz)',title=f'Spectrogram | {name} | {load}')
            fig.tight_layout();st.pyplot(fig);plt.close(fig)
        elif module=='Dominant Peaks':
            st.subheader('Dominant FFT Peaks')
            limit=st.slider('Maximum frequency (Hz)',500,6000,3000,250)
            f,amp=fft(x,limit);peaks=peak_frame(f,amp,12)
            fig,ax=plt.subplots(figsize=(12,4));ax.plot(f,amp,color=color,lw=.7)
            if not peaks.empty: ax.scatter(peaks['Frequency (Hz)'],peaks['Amplitude'],color='#102b49',s=20,zorder=4)
            ax.set(xlabel='Frequency (Hz)',ylabel='Amplitude',title=f'Dominant peaks | {name} | {load}')
            style_ax(ax);fig.tight_layout();st.pyplot(fig);plt.close(fig)
            st.dataframe(peaks,use_container_width=True,hide_index=True)
            download(peaks,'⬇ Export dominant peaks (CSV)','dominant_peaks.csv')
        elif module=='Envelope Spectrum':
            st.subheader('Envelope Spectrum')
            limit=st.slider('Maximum envelope frequency (Hz)',100,1000,300,50)
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
st.caption('CWRU bearing vibration dataset • 12 kHz drive-end signals • This dashboard supports exploratory signal analysis, not validated automatic fault classification.')
