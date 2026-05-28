import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import TremorCNN
from dataset import make_dataset, load_tremor_catalog
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import glob

# load tremor catalog
csv_files = glob.glob("tremor_events-*.csv")
print(f"Loading {len(csv_files)} catalog files...")
tremor_times = load_tremor_catalog(csv_files)
print(f"Total catalog events: {len(tremor_times)}")

client = Client("EARTHSCOPE")

def fetch(net, sta, t_start, t_end):
    for ch in ["BHZ", "HHZ", "EHZ"]:
        try:
            st = client.get_waveforms(net, sta, "--", ch,
                                      UTCDateTime(t_start), UTCDateTime(t_end))
            st.resample(40.0)
            return st[0].data, UTCDateTime(t_start)
        except:
            continue
    raise Exception(f"No data for {net}.{sta} {t_start}")


#fetch data from csv files of that content and labelling some file times and content/episodic tremor.

fetches = [
    # 2010 episode
    ("UW", "GNW",  "2010-08-15T00:00:00", "2010-08-15T06:00:00"),
    ("UW", "GNW",  "2010-08-20T00:00:00", "2010-08-20T06:00:00"),
    ("UW", "FORK", "2010-08-15T00:00:00", "2010-08-15T06:00:00"),
    ("UW", "LEBA", "2010-08-15T00:00:00", "2010-08-15T06:00:00"),
    # 2011 episode
    ("UW", "GNW",  "2011-08-10T00:00:00", "2011-08-10T06:00:00"),
    ("UW", "FORK", "2011-08-10T00:00:00", "2011-08-10T06:00:00"),
    ("UW", "LEBA", "2011-08-10T00:00:00", "2011-08-10T06:00:00"),
    # 2012 episode
    ("UW", "GNW",  "2012-09-20T00:00:00", "2012-09-20T06:00:00"),
    ("UW", "FORK", "2012-09-20T00:00:00", "2012-09-20T06:00:00"),
    # 2013 episode
    ("UW", "GNW",  "2013-09-20T00:00:00", "2013-09-20T06:00:00"),
    ("UW", "FORK", "2013-09-20T00:00:00", "2013-09-20T06:00:00"),
    # 2014 episode
    ("UW", "GNW",  "2014-11-15T00:00:00", "2014-11-15T06:00:00"),
    # 2015-16 episode
    ("UW", "GNW",  "2016-01-05T00:00:00", "2016-01-05T06:00:00"),
    # 2017 episode
    ("UW", "GNW",  "2017-02-25T00:00:00", "2017-02-25T06:00:00"),
    # 2018 episode
    ("UW", "GNW",  "2018-06-01T00:00:00", "2018-06-01T06:00:00"),
    # quiet periods
    ("UW", "GNW",  "2010-03-01T00:00:00", "2010-03-01T06:00:00"),
    ("UW", "GNW",  "2010-03-01T12:00:00", "2010-03-01T18:00:00"),
    ("UW", "GNW",  "2011-04-01T00:00:00", "2011-04-01T06:00:00"),
    ("UW", "GNW",  "2011-04-01T12:00:00", "2011-04-01T18:00:00"),
    ("UW", "GNW",  "2012-06-01T00:00:00", "2012-06-01T06:00:00"),
    ("UW", "GNW",  "2012-06-01T12:00:00", "2012-06-01T18:00:00"),
    ("UW", "GNW",  "2014-06-01T00:00:00", "2014-06-01T06:00:00"),
    ("UW", "GNW",  "2014-06-01T12:00:00", "2014-06-01T18:00:00"),
    ("UW", "FORK", "2014-06-01T00:00:00", "2014-06-01T06:00:00"),
    ("UW", "GNW",  "2015-06-01T00:00:00", "2015-06-01T06:00:00"),
    ("UW", "GNW",  "2015-06-01T12:00:00", "2015-06-01T18:00:00"),
    ("UW", "GNW",  "2016-06-01T00:00:00", "2016-06-01T06:00:00"),
    ("UW", "GNW",  "2019-06-01T00:00:00", "2019-06-01T06:00:00"),
    ("UW", "GNW",  "2019-06-01T12:00:00", "2019-06-01T18:00:00"),
    ("UW", "FORK", "2019-06-01T00:00:00", "2019-06-01T06:00:00"),
] 

waveforms = []
starts    = []

for net, sta, t_start, t_end in fetches: #If you have obtained the files from my prior commit, this is just for checking if the files/data is available in device to train.
    try:
        data, t0 = fetch(net, sta, t_start, t_end)
        waveforms.append(data)
        starts.append(t0)
        print(f"OK  {net}.{sta} {t_start[:10]}")
    except Exception as e:
        print(f"SKIP {net}.{sta} {t_start[:10]} — {e}")


X, y, timestamps = make_dataset(waveforms, starts, tremor_times)
print(f"Windows: {X.shape}")
print(f"Tremor: {y.sum()}, Quiet: {(y==0).sum()}")

# Chronologically sorted, taken inspo from the paper.
sort_idx   = np.argsort(timestamps)
X          = X[sort_idx]
y          = y[sort_idx]
timestamps = timestamps[sort_idx]

# --- chronological split ---
split = int(0.8 * len(X))
print(f"Train up to: {UTCDateTime(timestamps[split-1])}")
print(f"Test  from:  {UTCDateTime(timestamps[split])}")

# --- normalize on train stats only ---
X_log   = 10 * np.log10(X + 1e-10)
X_train = X_log[:split]
X_test  = X_log[split:]
mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

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
model      = TremorCNN().to(device)
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
n_quiet    = int((y==0).sum())
n_tremor   = int(y.sum())
pos_weight = torch.tensor([n_quiet / n_tremor]).to(device)
print(f"pos_weight: {n_quiet/n_tremor:.2f}")
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
EPOCHS     = 20
losses     = []

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
all_probs  = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        prob = torch.sigmoid(model(xb).squeeze())
        pred = (prob > 0.7).float()
        tp += ((pred == 1) & (yb == 1)).sum().item()
        fp += ((pred == 1) & (yb == 0)).sum().item()
        tn += ((pred == 0) & (yb == 0)).sum().item()
        fn += ((pred == 0) & (yb == 1)).sum().item()
        all_probs.extend(prob.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

correct   = tp + tn
precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)
auc       = roc_auc_score(all_labels, all_probs)

print(f"\nTest accuracy: {correct}/{len(test_ds)} = {correct/len(test_ds):.1%}")
print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
print(f"ROC-AUC:   {auc:.3f}  (paper reported 0.945)")
print(f"True Positives  (tremor correctly detected):  {tp}")
print(f"False Positives (quiet incorrectly flagged):  {fp}")
print(f"True Negatives  (quiet correctly identified): {tn}")
print(f"False Negatives (tremor missed):              {fn}")

torch.save(model.state_dict(), "tremor_cnn.pth")
print("Model saved to tremor_cnn.pth")

plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss - TremorCNN (catalog-labeled)')
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
print("Loss curve saved to loss_curve.png")