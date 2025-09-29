# src/kmnist/data.py
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# ---------- Load raw KMNIST ----------
def _npz(path: Path, key="arr_0"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)[key]

def load_kmnist_npz(data_dir: Path, split: str):
    """
    Load KMNIST from local .npz dumps.
    split: "train" or "test"
    Returns: (images_uint8 [N,28,28], labels_int64 [N])
    """
    data_dir = Path(data_dir)
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    imgs = _npz(data_dir / f"kmnist-{split}-imgs.npz")   # uint8 [N,28,28]
    labs = _npz(data_dir / f"kmnist-{split}-labels.npz") # int64 [N]
    return imgs, labs

# ---------- Split train/val ----------
def split_train_val(x, y, val_frac: float = 0.1, seed: int = 42):
    """
    Split arrays into train/val and return:
    (x_train, y_train, x_val, y_val)
    """
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_frac))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]

# ---------- Build DataLoader ----------
def make_loader(x: np.ndarray, y: np.ndarray, batch_size=128, shuffle=True):
    """
    Convert numpy arrays to a DataLoader.
    x: (N,28,28) uint8 -> float32 [0,1] inverted (white-on-black)
    y: (N,) int64
    """
    x = x.astype(np.float32) / 255.0
    x = 1.0 - x                       # invert: white on black
    x = torch.from_numpy(x)[:, None]  # -> (N,1,28,28)
    y = torch.from_numpy(y.astype(np.int64))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)