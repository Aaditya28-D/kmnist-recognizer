from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

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

def make_loader(x_u8: np.ndarray, y: np.ndarray, batch_size=128, shuffle=True):
    """
    Convert uint8 images [N,28,28] -> float32 tensor in [0,1] with white-on-black,
    and build a DataLoader.
    """
    x = torch.from_numpy(x_u8.astype("float32") / 255.0)
    x = 1.0 - x  # invert: white ink on black bg like your app
    x = x.unsqueeze(1)  # [N,1,28,28]
    y = torch.from_numpy(y.astype("int64"))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def split_train_val(x, y, val_frac: float = 0.1, seed: int = 42):
    """
    Split arrays into train/val and return FOUR arrays:
    (tr_x, tr_y, va_x, va_y)

    x: numpy array of images  (N, 28, 28) or (N, H, W)
    y: numpy array of labels  (N,)
    """
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_frac))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def make_loader(x, y, batch_size=128, shuffle=True):
    """
    Convert numpy arrays to a torch DataLoader.
    x: (N, 28, 28) uint8  -> normalized float32 [0,1] and inverted to white-on-black
    y: (N,) int
    """
    x = x.astype(np.float32) / 255.0
    x = 1.0 - x                       # match your training convention (white on black)
    x = torch.from_numpy(x)[:, None]  # (N,1,28,28)
    y = torch.from_numpy(y.astype(np.int64))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)