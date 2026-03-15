import numpy as np
import json, os
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import h5py

NPZ_PATH = '../Data.npz'
DATA_DIR = '../Data'

print("Загрузка данных...")

if os.path.exists(NPZ_PATH):
    data = np.load(NPZ_PATH, allow_pickle=True)
    print(f'Ключи в NPZ: {list(data.files)}')
    train_x, train_y_raw = data['train_x'], data['train_y']
    valid_x, valid_y_raw = data['valid_x'], data['valid_y']
else:
    train_x = np.load(f'{DATA_DIR}/train_x.npy', allow_pickle=True)
    train_y_raw = np.load(f'{DATA_DIR}/train_y.npy', allow_pickle=True)
    valid_x = np.load(f'{DATA_DIR}/valid_x.npy', allow_pickle=True)
    valid_y_raw = np.load(f'{DATA_DIR}/valid_y.npy', allow_pickle=True)

print(f'train_x: {train_x.shape}, train_y[0]: {train_y_raw[0]}')

print("Восстановление меток...")
train_names = [str(s)[32:] for s in train_y_raw]
valid_names = [str(s)[32:] for s in valid_y_raw]

all_classes = sorted(set(train_names))
label_map = {name: i for i, name in enumerate(all_classes)}
label_map_inv = {v: k for k, v in label_map.items()}
num_classes = len(all_classes)

y_train = np.array([label_map[n] for n in train_names])
y_val = np.array([label_map[n] for n in valid_names])
print(f'{num_classes} классов: {all_classes}')

print("Извлечение признаков (мел-спектрограммы)...")

SR = 16000
N_MELS = 64
N_FFT = 1024
HOP = 512
FIXED_FRAMES = 160

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(SR / 2), N_MELS + 2)
hz_pts = mel_to_hz(mel_pts)
bin_pts = np.floor((N_FFT + 1) * hz_pts / SR).astype(int)
MEL_FB = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
for m in range(1, N_MELS + 1):
    lo, mid, hi = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
    for k in range(lo, mid):
        if mid > lo: MEL_FB[m-1, k] = (k - lo) / (mid - lo)
    for k in range(mid, hi):
        if hi > mid: MEL_FB[m-1, k] = (hi - k) / (hi - mid)

WINDOW = np.hanning(N_FFT).astype(np.float32)

def audio_to_melspec(audio_arr):
    arr = audio_arr.squeeze().astype(np.float32)
    peak = np.abs(arr).max()
    if peak > 0:
        arr = arr / peak
    n = len(arr)
    n_frames = min(max(1, (n - N_FFT) // HOP + 1), FIXED_FRAMES)
    padded = np.zeros(n_frames * HOP + N_FFT, dtype=np.float32)
    padded[:n] = arr
    shape = (n_frames, N_FFT)
    strides = (padded.strides[0] * HOP, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides) * WINDOW
    spec = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2
    mel = np.log1p(np.dot(spec, MEL_FB.T)).T
    if mel.shape[1] < FIXED_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, FIXED_FRAMES - mel.shape[1])))
    return mel[:, :FIXED_FRAMES].astype(np.float32)

X_train = np.array([audio_to_melspec(train_x[i]) for i in range(len(train_x))])
X_val = np.array([audio_to_melspec(valid_x[i]) for i in range(len(valid_x))])
print(f'X_train: {X_train.shape}, X_val: {X_val.shape}')

spec_mean = float(X_train.mean())
spec_std = float(X_train.std())
X_train_n = (X_train - spec_mean) / spec_std
X_val_n = (X_val - spec_mean) / spec_std

os.makedirs('../model', exist_ok=True)
with open('../model/spec_norm.json', 'w') as f:
    json.dump({'mean': spec_mean, 'std': spec_std, 'sr': SR,
               'n_mels': N_MELS, 'n_fft': N_FFT, 'hop': HOP,
               'fixed_frames': FIXED_FRAMES}, f)

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = AudioCNN(num_classes)

print("Обучение модели...")
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3

X_tr = torch.tensor(X_train_n).unsqueeze(1)
y_tr = torch.tensor(y_train, dtype=torch.long)
X_vl = torch.tensor(X_val_n).unsqueeze(1)
y_vl = torch.tensor(y_val, dtype=torch.long)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.CrossEntropyLoss()

log = []
best_acc = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    tot_loss = tot_n = tot_ok = 0
    for xb, yb in loader:
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        opt.step()
        tot_loss += loss.item() * len(xb)
        tot_ok += (out.argmax(1) == yb).sum().item()
        tot_n += len(xb)
    train_loss = tot_loss / tot_n
    train_acc = tot_ok / tot_n

    model.eval()
    with torch.no_grad():
        vout = model(X_vl)
        val_loss = criterion(vout, y_vl).item()
        val_acc = (vout.argmax(1) == y_vl).float().mean().item()
    sched.step()

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    log.append(dict(epoch=epoch,
                    train_loss=round(train_loss, 5), train_accuracy=round(train_acc, 5),
                    val_loss=round(val_loss, 5), val_accuracy=round(val_acc, 5)))
    if epoch % 10 == 0 or epoch == 1:
        print(f'Эпоха {epoch:>3}: потери={train_loss:.4f} точн={train_acc*100:.1f}% | val точн={val_acc*100:.1f}%')

print(f'\nЛучшая точность на валидации: {best_acc*100:.2f}%')

if best_state:
    model.load_state_dict(best_state)

torch.save(model.state_dict(), '../model/model.pt')

# Save model in .h5 format (HDF5) for submission requirements
def save_model_h5(model, filepath, label_map, spec_norm):
    """Save PyTorch model weights to HDF5 format."""
    state_dict = model.state_dict()
    with h5py.File(filepath, 'w') as f:
        # Save model weights
        weights_grp = f.create_group('model_weights')
        for name, param in state_dict.items():
            weights_grp.create_dataset(name, data=param.numpy())
        
        # Save metadata
        meta_grp = f.create_group('metadata')
        meta_grp.create_dataset('num_classes', data=num_classes)
        meta_grp.attrs['label_map'] = json.dumps(label_map)
        meta_grp.attrs['spec_norm'] = json.dumps(spec_norm)
        meta_grp.attrs['architecture'] = 'AudioCNN'

save_model_h5(model, '../model/model.h5', label_map, 
              {'mean': spec_mean, 'std': spec_std, 'sr': SR,
               'n_mels': N_MELS, 'n_fft': N_FFT, 'hop': HOP,
               'fixed_frames': FIXED_FRAMES})

with open('../model/training_log.json', 'w') as f:
    json.dump(log, f, indent=2)
with open('../model/label_map.json', 'w') as f:
    json.dump(label_map, f, indent=2)

dist = Counter(y_train.tolist())
with open('../model/class_distribution.json', 'w') as f:
    json.dump({label_map_inv[i]: dist[i] for i in range(num_classes)}, f, indent=2)

print("Сохранено: model.pt, model.h5, training_log.json, label_map.json, spec_norm.json, class_distribution.json")

eps = [e['epoch'] for e in log]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(eps, [e['train_accuracy'] for e in log], label='Обучение')
ax1.plot(eps, [e['val_accuracy'] for e in log], label='Валидация')
ax1.set(xlabel='Эпоха', ylabel='Точность', title='Точность по эпохам')
ax1.legend(); ax1.grid(True)
ax2.plot(eps, [e['train_loss'] for e in log], label='Обучение')
ax2.plot(eps, [e['val_loss'] for e in log], label='Валидация')
ax2.set(xlabel='Эпоха', ylabel='Потери', title='Потери по эпохам')
ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig('../model/training_curves.png', dpi=150)
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
model.eval()
with torch.no_grad():
    val_preds = model(X_vl).argmax(1).numpy()
cm = confusion_matrix(y_val, val_preds)
cls_names = [label_map_inv[i] for i in range(num_classes)]
fig, ax = plt.subplots(figsize=(14, 12))
ConfusionMatrixDisplay(cm, display_labels=cls_names).plot(ax=ax, xticks_rotation=45)
ax.set_title('Матрица ошибок (валидация)')
plt.tight_layout()
plt.savefig('../model/confusion_matrix.png', dpi=150)
plt.show()

print(f'Итоговая точность на валидации: {(val_preds == y_val).mean()*100:.2f}%')
print('Готово.')
