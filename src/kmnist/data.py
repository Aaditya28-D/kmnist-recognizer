# src/kmnist/data.py
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_KX = None
_KY = None


def load_kmnist_test():
    """
    Loads KMNIST test set from:
      data/kmnist-test-imgs.npz
      data/kmnist-test-labels.npz

    Caches in memory after first load.
    """
    global _KX, _KY

    if _KX is not None:
        return _KX, _KY

    imgs = ROOT / "data/kmnist-test-imgs.npz"
    labs = ROOT / "data/kmnist-test-labels.npz"

    _KX = np.load(imgs)["arr_0"]  # (10000, 28, 28) uint8
    _KY = np.load(labs)["arr_0"]  # (10000,) int
    return _KX, _KY