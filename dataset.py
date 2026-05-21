import numpy as np
from scipy.signal import spectrogram
from obspy.signal.filter import bandpass
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

WINDOW_SEC   = 60
STEP_SEC     = 30
FS           = 40.0
NPERSEG      = 256
TREMOR_START = 1800
TREMOR_END   = 3600

def make_dataset(quiet_list, tremor_list, quiet_starts, tremor_starts):
    """
    quiet_list     : list of numpy arrays (raw waveforms)
    tremor_list    : list of numpy arrays (raw waveforms)
    quiet_starts   : list of UTCDateTime — start time of each quiet fetch
    tremor_starts  : list of UTCDateTime — start time of each tremor fetch
    
    returns X, y, timestamps (as float unix seconds)
    """
    windows    = []
    labels     = []
    timestamps = []

    for data_quiet, t0 in zip(quiet_list, quiet_starts):
        for start in range(0, len(data_quiet) - int(WINDOW_SEC * FS), int(STEP_SEC * FS)):
            raw_chunk = data_quiet[start : start + int(WINDOW_SEC * FS)]
            chunk = bandpass(raw_chunk.astype(float), freqmin=1.0, freqmax=8.0, df=FS, corners=4)
            f, t, Sxx = spectrogram(chunk, FS, nperseg=NPERSEG)
            freq_mask = (f >= 1) & (f <= 8)
            Sxx = Sxx[freq_mask]
            windows.append(Sxx)
            labels.append(0)
            timestamps.append(float(t0) + start / FS)   # unix timestamp of window start

    for data_tremor, t0 in zip(tremor_list, tremor_starts):
        tremor_chunk = data_tremor[int(TREMOR_START * FS) : int(TREMOR_END * FS)]
        for start in range(0, len(tremor_chunk) - int(WINDOW_SEC * FS), int(STEP_SEC * FS)):
            raw_chunk = tremor_chunk[start : start + int(WINDOW_SEC * FS)]
            chunk = bandpass(raw_chunk.astype(float), freqmin=1.0, freqmax=8.0, df=FS, corners=4)
            f, t, Sxx = spectrogram(chunk, FS, nperseg=NPERSEG)
            freq_mask = (f >= 1) & (f <= 8)
            Sxx = Sxx[freq_mask]
            windows.append(Sxx)
            labels.append(1)
            timestamps.append(float(t0) + TREMOR_START + start / FS)

    return np.array(windows), np.array(labels), np.array(timestamps)

if __name__ == "__main__":
    client = Client("EARTHSCOPE")
    t_quiet  = UTCDateTime("2010-08-05T23:00:00")
    t_tremor = UTCDateTime("2010-08-06T00:00:00")
    st_quiet  = client.get_waveforms("UW", "GNW", "--", "BHZ", t_quiet,  t_quiet  + 3600)
    st_tremor = client.get_waveforms("UW", "GNW", "--", "BHZ", t_tremor, t_tremor + 3600)
    st_quiet[0].resample(40.0)
    st_tremor[0].resample(40.0)
    X, y, ts = make_dataset([st_quiet[0].data], [st_tremor[0].data], [t_quiet], [t_tremor])
    print(f"Windows: {X.shape}, Labels: {y.shape}, Timestamps: {ts.shape}")
    print(f"First timestamp: {UTCDateTime(ts[0])}")
    print(f"Last timestamp:  {UTCDateTime(ts[-1])}")