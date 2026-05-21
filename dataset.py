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

def make_dataset(quiet_list, tremor_list):
    windows = []
    labels  = []

    for data_quiet in quiet_list:
        for start in range(0, len(data_quiet) - int(WINDOW_SEC * FS), int(STEP_SEC * FS)):
            raw_chunk = data_quiet[start : start + int(WINDOW_SEC * FS)]
            chunk = bandpass(raw_chunk.astype(float), freqmin=1.0, freqmax=8.0, df=FS, corners=4)
            f, t, Sxx = spectrogram(chunk, FS, nperseg=NPERSEG)
            freq_mask = (f >= 1) & (f <= 8)
            Sxx = Sxx[freq_mask]
            windows.append(Sxx)
            labels.append(0)

    for data_tremor in tremor_list:
        tremor_chunk = data_tremor[int(TREMOR_START * FS) : int(TREMOR_END * FS)]
        for start in range(0, len(tremor_chunk) - int(WINDOW_SEC * FS), int(STEP_SEC * FS)):
            raw_chunk = tremor_chunk[start : start + int(WINDOW_SEC * FS)]
            chunk = bandpass(raw_chunk.astype(float), freqmin=1.0, freqmax=8.0, df=FS, corners=4)
            f, t, Sxx = spectrogram(chunk, FS, nperseg=NPERSEG)
            freq_mask = (f >= 1) & (f <= 8)
            Sxx = Sxx[freq_mask]
            windows.append(Sxx)
            labels.append(1)

    return np.array(windows), np.array(labels)

if __name__ == "__main__":

    client = Client("EARTHSCOPE")
    st_quiet  = client.get_waveforms("UW", "GNW", "--", "BHZ",
                    UTCDateTime("2010-08-05T23:00:00"),
                    UTCDateTime("2010-08-06T00:00:00"))
    st_tremor = client.get_waveforms("UW", "GNW", "--", "BHZ",
                    UTCDateTime("2010-08-06T00:00:00"),
                    UTCDateTime("2010-08-06T01:00:00"))

    data_quiet  = st_quiet[0].data
    data_tremor = st_tremor[0].data

    X, y = make_dataset([data_quiet], [data_tremor])
    print(f"Windows: {X.shape}")
    print(f"Labels:  {y.shape}")
    print(f"Tremor windows: {y.sum()}, Quiet windows: {(y==0).sum()}")