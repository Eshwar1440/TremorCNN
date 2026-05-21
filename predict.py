import torch
import numpy as np
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.signal.filter import bandpass
from scipy.signal import spectrogram
from model import TremorCNN

FS         = 40.0
NPERSEG    = 256
WINDOW_SEC = 60
STEP_SEC   = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load trained model ---
model = TremorCNN().to(device)
model.load_state_dict(torch.load("tremor_cnn.pth", map_location=device))
model.eval()
print(f"Model loaded | device: {device}")

# --- fetch unseen station + unseen episode ---
client = Client("EARTHSCOPE")
print("Fetching UW.MRBL Sep 15 2011 (never seen in training)...")
st = client.get_waveforms("UW", "MRBL", "--", "BHZ",
        UTCDateTime("2011-09-15T00:00:00"),
        UTCDateTime("2011-09-15T01:00:00"))
st.resample(40.0)
data = st[0].data

# --- sliding window prediction ---
times = []
probs = []

with torch.no_grad():
    for start in range(0, len(data) - int(WINDOW_SEC * FS), int(STEP_SEC * FS)):
        raw   = data[start : start + int(WINDOW_SEC * FS)]
        chunk = bandpass(raw.astype(float), freqmin=1.0, freqmax=8.0, df=FS, corners=4)
        f, t, Sxx = spectrogram(chunk, FS, nperseg=NPERSEG)
        mask  = (f >= 1) & (f <= 8)
        Sxx   = Sxx[mask]
        Sxx   = 10 * np.log10(Sxx + 1e-10)
        Sxx   = (Sxx - Sxx.mean()) / Sxx.std()
        x     = torch.tensor(Sxx, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        prob  = model(x).item()
        times.append(start / FS / 60)
        probs.append(prob)

times = np.array(times)
probs = np.array(probs)

# --- print raw numbers ---
print("\nTime(min) | P(tremor) | Decision")
print("-" * 40)
for t, p in zip(times, probs):
    decision = "TREMOR" if p > 0.5 else "quiet"
    print(f"{t:6.1f}    | {p:.4f}    | {decision}")

# --- plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))

t_wave = np.linspace(0, 60, len(data))
ax1.plot(t_wave, data, color='black', linewidth=0.3)
ax1.set_ylabel('Ground velocity (counts)')
ax1.set_title('UW.MRBL - Sep 15 2011')
ax1.set_xlim(0, 60)

ax2.plot(times, probs, color='red', linewidth=2, label='Tremor probability')
ax2.axhline(0.5, color='black', linestyle='--', linewidth=0.8, label='Decision threshold (0.5)')
ax2.fill_between(times, probs, 0.5,
                  where=(probs >= 0.5), color='red', alpha=0.3, label='Tremor detected')
ax2.fill_between(times, probs, 0.5,
                  where=(probs < 0.5), color='green', alpha=0.3, label='Quiet detected')
ax2.set_ylim(0, 1)
ax2.set_xlim(0, 60)
ax2.set_xlabel('Time (minutes)')
ax2.set_ylabel('P(tremor)')
ax2.set_title('CNN Tremor Probability')
ax2.legend()

plt.tight_layout()
plt.savefig('prediction_timeline.png', dpi=150, bbox_inches='tight')
print("\nSaved prediction_timeline.png")