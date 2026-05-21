import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from model import TremorCNN
from dataset import make_dataset
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt

client = Client("EARTHSCOPE")

def fetch(net, sta, t_start, t_end):
    st = client.get_waveforms(net, sta, "--", "BHZ",
                              UTCDateTime(t_start), UTCDateTime(t_end))
    st.resample(40.0)
    return st[0].data, UTCDateTime(t_start)

print("Fetching data...")

# --- QUIET ---
quiet_data, quiet_times = zip(*[
    fetch("UW", "GNW", "2010-01-15T00:00:00", "2010-01-15T02:00:00"),
    fetch("UW", "GNW", "2010-01-15T06:00:00", "2010-01-15T08:00:00"),
    fetch("UW", "GNW", "2010-01-15T12:00:00", "2010-01-15T14:00:00"),
    fetch("UW", "GNW", "2010-01-15T18:00:00", "2010-01-15T20:00:00"),
    fetch("UW", "GNW", "2012-03-15T00:00:00", "2012-03-15T02:00:00"),
    fetch("UW", "GNW", "2012-03-15T06:00:00", "2012-03-15T08:00:00"),
    fetch("UW", "GNW", "2012-03-15T12:00:00", "2012-03-15T14:00:00"),
    fetch("UW", "GNW", "2012-03-15T18:00:00", "2012-03-15T20:00:00"),
    fetch("UW", "GNW",  "2013-01-15T00:00:00", "2013-01-15T02:00:00"),
    fetch("UW", "GNW",  "2013-01-15T12:00:00", "2013-01-15T14:00:00"),
    fetch("UW", "FORK", "2013-01-15T00:00:00", "2013-01-15T02:00:00"),
    fetch("UW", "FORK", "2013-01-15T12:00:00", "2013-01-15T14:00:00"),
    fetch("UW", "GNW", "2010-08-05T23:00:00", "2010-08-06T00:00:00"),
    fetch("UW", "GNW", "2010-08-04T00:00:00", "2010-08-04T02:00:00"),
    fetch("UW", "GNW", "2010-08-04T12:00:00", "2010-08-04T14:00:00"),
])

# --- TREMOR ---
tremor_data, tremor_times = zip(*[
    fetch("UW", "GNW",  "2010-08-06T00:00:00", "2010-08-06T01:00:00"),
    fetch("UW", "GNW",  "2010-08-08T00:00:00", "2010-08-08T02:00:00"),
    fetch("UW", "FORK", "2010-08-06T00:00:00", "2010-08-06T01:00:00"),
    fetch("UW", "LEBA", "2010-08-06T00:00:00", "2010-08-06T01:00:00"),
    fetch("UW", "GNW",  "2011-09-15T00:00:00", "2011-09-15T02:00:00"),
    fetch("UW", "LEBA", "2011-09-15T00:00:00", "2011-09-15T02:00:00"),
    fetch("UW", "FORK", "2011-09-15T00:00:00", "2011-09-15T02:00:00"),
    fetch("UW", "GNW",  "2013-10-01T00:00:00", "2013-10-01T02:00:00"),
    fetch("UW", "FORK", "2013-10-01T00:00:00", "2013-10-01T02:00:00"),
])

print("Building dataset...")
X, y, timestamps = make_dataset(
    list(quiet_data), list(tremor_data),
    list(quiet_times), list(tremor_times)
)
print(f"Windows: {X.shape}, Tremor: {y.sum()}, Quiet: {(y==0).sum()}")


sort_idx   = np.argsort(timestamps)
X          = X[sort_idx]
y          = y[sort_idx]
timestamps = timestamps[sort_idx]

#80% train, 20% test
split      = int(0.8 * len(X))
print(f"Train up to: {UTCDateTime(timestamps[split-1])}")
print(f"Test from:   {UTCDateTime(timestamps[split])}")

# --- normalize ---
X_train = 10 * np.log10(X[:split] + 1e-10)
X_test  = 10 * np.log10(X[split:] + 1e-10)
mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std   # use train stats, not test stats

# --- PyTorch Dataset ---
class SeismicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_ds = SeismicDataset(X_train, y[:split])
test_ds  = SeismicDataset(X_test,  y[split:])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- train ---
model     = TremorCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

EPOCHS = 20
losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch+1}/{EPOCHS}  loss: {losses[-1]:.4f}")

# --- evaluate ---
model.eval()
tp = fp = tn = fn = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = (model(xb).squeeze() > 0.5).float()
        tp += ((pred == 1) & (yb == 1)).sum().item()
        fp += ((pred == 1) & (yb == 0)).sum().item()
        tn += ((pred == 0) & (yb == 0)).sum().item()
        fn += ((pred == 0) & (yb == 1)).sum().item()

correct   = tp + tn
precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)

print(f"\nTest accuracy: {correct}/{len(test_ds)} = {correct/len(test_ds):.1%}")
print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
print(f"True Positives  (tremor correctly detected):       {tp}")
print(f"False Positives (quiet incorrectly flagged):       {fp}")
print(f"True Negatives  (quiet correctly identified):      {tn}")
print(f"False Negatives (tremor missed):                   {fn}")

torch.save(model.state_dict(), 'tremor_cnn.pth')
print("Model saved to tremor_cnn.pth")

plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss - TremorCNN (chronological split)')
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
print("Loss curve saved to loss_curve.png")