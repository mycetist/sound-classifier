import torch
import torch.nn as nn
import numpy as np
import json
import os
import h5py

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')

class AudioCNN(nn.Module):
    def __init__(self, num_classes=20):
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

_model = None
_spec_norm = None
_label_map = None
_label_map_inv = None
_mel_fb = None  # cached mel filterbank

def _build_mel_filterbank(sr, n_fft, n_mels):
    def hz2mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel2hz(m):  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mel_pts  = np.linspace(hz2mel(0), hz2mel(sr / 2.0), n_mels + 2)
    hz_pts   = mel2hz(mel_pts)
    bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, mid, hi = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
        for k in range(lo, mid):
            if mid > lo: fb[m-1, k] = (k - lo) / (mid - lo)
        for k in range(mid, hi):
            if hi > mid: fb[m-1, k] = (hi - k) / (hi - mid)
    return fb

def _audio_to_melspec(audio_arr, norm):
    global _mel_fb
    sr    = norm.get('sr', 16000)
    n_mels= norm.get('n_mels', 64)
    n_fft = norm.get('n_fft', 1024)
    hop   = norm.get('hop', 512)
    fixed = norm.get('fixed_frames', 160)
    mean  = norm.get('mean', 0.0)
    std   = norm.get('std', 1.0)

    if _mel_fb is None:
        _mel_fb = _build_mel_filterbank(sr, n_fft, n_mels)

    if isinstance(audio_arr, (bytes, bytearray)):
        import io, scipy.io.wavfile as wav
        rate, data = wav.read(io.BytesIO(audio_arr))
        arr = data.astype(np.float32)
        peak = np.abs(arr).max()
        if peak > 0: arr = arr / peak
    else:
        arr = np.asarray(audio_arr).squeeze().astype(np.float32)
        peak = np.abs(arr).max()
        if peak > 0: arr = arr / peak

    n = len(arr)
    n_frames = max(1, min((n - n_fft) // hop + 1, fixed))
    window = np.hanning(n_fft).astype(np.float32)
    padded = np.zeros(n_frames * hop + n_fft, dtype=np.float32)
    padded[:n] = arr
    shape = (n_frames, n_fft)
    strides = (padded.strides[0] * hop, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides) * window

    spec   = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2   # (n_frames, n_fft//2+1)
    mel    = np.log1p(np.dot(spec, _mel_fb.T)).T          # (n_mels, n_frames)

    if mel.shape[1] < fixed:
        mel = np.pad(mel, ((0, 0), (0, fixed - mel.shape[1])))
    else:
        mel = mel[:, :fixed]

    return ((mel - mean) / std).astype(np.float32)        # (n_mels, fixed)

# ── Load model ────────────────────────────────────────────────────────────────
def _load_weights_h5(filepath, model):
    """Load model weights from HDF5 (.h5) file."""
    state_dict = {}
    with h5py.File(filepath, 'r') as f:
        weights_grp = f['model_weights']
        for name in weights_grp.keys():
            state_dict[name] = torch.tensor(weights_grp[name][:])
    model.load_state_dict(state_dict)
    return model

def load_model():
    global _model, _spec_norm, _label_map, _label_map_inv
    if _label_map is None:
        with open(os.path.join(MODEL_DIR, 'label_map.json')) as f:
            _label_map = json.load(f)
        _label_map_inv = {v: k for k, v in _label_map.items()}
    if _spec_norm is None:
        p = os.path.join(MODEL_DIR, 'spec_norm.json')
        if os.path.exists(p):
            with open(p) as f:
                _spec_norm = json.load(f)
        else:
            _spec_norm = {}
    if _model is None:
        _model = AudioCNN(num_classes=len(_label_map))
        # Try .pt first, then .h5
        pt_path = os.path.join(MODEL_DIR, 'model.pt')
        h5_path = os.path.join(MODEL_DIR, 'model.h5')
        if os.path.exists(pt_path):
            _model.load_state_dict(torch.load(pt_path, map_location='cpu', weights_only=True))
        elif os.path.exists(h5_path):
            _model = _load_weights_h5(h5_path, _model)
        else:
            raise FileNotFoundError(f"No model found at {pt_path} or {h5_path}")
        _model.eval()
    return _model, _spec_norm, _label_map, _label_map_inv

def predict(audio_arr):
    model, norm, label_map, label_map_inv = load_model()
    spec   = _audio_to_melspec(audio_arr, norm)
    tensor = torch.tensor(spec).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,fixed)
    with torch.no_grad():
        probs      = torch.softmax(model(tensor), dim=1)
        pred_class = probs.argmax(1).item()
        confidence = probs[0, pred_class].item()
    return label_map_inv[pred_class], pred_class, confidence

def predict_batch(audio_list):
    model, norm, label_map, label_map_inv = load_model()
    specs  = np.stack([_audio_to_melspec(a, norm) for a in audio_list])
    tensor = torch.tensor(specs).unsqueeze(1)              # (N,1,n_mels,fixed)
    with torch.no_grad():
        probs        = torch.softmax(model(tensor), dim=1)
        pred_classes = probs.argmax(1).tolist()
        confidences  = [probs[i, c].item() for i, c in enumerate(pred_classes)]
    return [{'class_name': label_map_inv[c], 'class_id': c, 'confidence': conf}
            for c, conf in zip(pred_classes, confidences)]

def process_audio_batch(audio_list):
    """Legacy compatibility — predict_batch now handles everything."""
    return audio_list

def get_training_log():
    with open(os.path.join(MODEL_DIR, 'training_log.json')) as f:
        return json.load(f)

def get_class_distribution():
    p = os.path.join(MODEL_DIR, 'class_distribution.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    with open(os.path.join(MODEL_DIR, 'label_map.json')) as f:
        lm = json.load(f)
    return {name: 1200 // len(lm) for name in lm}
