import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from obspy.signal.filter import bandpass
from obspy import UTCDateTime

WINDOW_SEC   = 60
STEP_SEC     = 30
FS           = 40.0
NPERSEG      = 256

def load_tremor_catalog(csv_paths):
    dfs = []
    for path in csv_paths: #Dont worry, the files I used are in one of my prior commits.
        df = pd.read_csv(path, skipinitialspace=True)
        dfs.append(df)
    catalog = pd.concat(dfs, ignore_index=True)
    catalog['starttime'] = pd.to_datetime(catalog['starttime'])
    catalog = catalog.sort_values('starttime').reset_index(drop=True)
    
    times = np.array([t.timestamp() for t in catalog['starttime']])
    return times

def is_tremor_window(window_start_unix, window_end_unix, tremor_times, tolerance=300):
   
    mask = (tremor_times >= window_start_unix - tolerance) & \
           (tremor_times <= window_end_unix + tolerance)
    return mask.any()

def make_dataset(waveform_list, start_times, tremor_catalog_times):
    """
    waveform_list     : list of numpy arrays (raw waveforms at FS=40Hz)
    start_times       : list of UTCDateTime — start time of each waveform fetch
    tremor_catalog_times : numpy array of unix timestamps from PNSN catalog

    Returns X (windows), y (labels), timestamps
    """
    windows    = []
    labels     = []
    timestamps = []

    for data, t0 in zip(waveform_list, start_times):
        t0_unix = float(t0)
        n_samples = len(data)

        for start in range(0, n_samples - int(WINDOW_SEC * FS), int(STEP_SEC * FS)):
            raw_chunk = data[start : start + int(WINDOW_SEC * FS)]
            chunk = bandpass(raw_chunk.astype(float),
                             freqmin=1.0, freqmax=8.0, df=FS, corners=4)
            f, t, Sxx = spectrogram(chunk, FS, nperseg=NPERSEG)
            freq_mask = (f >= 1) & (f <= 8)
            Sxx = Sxx[freq_mask]

            win_start = t0_unix + start / FS
            win_end   = win_start + WINDOW_SEC
            label = 1 if is_tremor_window(win_start, win_end,
                                           tremor_catalog_times) else 0

            windows.append(Sxx)
            labels.append(label)
            timestamps.append(win_start)

    return np.array(windows), np.array(labels), np.array(timestamps)