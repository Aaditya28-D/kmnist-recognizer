# src/kmnist/infer.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from .models.mlp import MLPWide  # add more architectures here if needed


# ------------------------- device -------------------------
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ------------------------- arch builder -------------------------
def detect_arch_from_filename(p: Path) -> str:
    """Very simple filename -> arch rule. Extend if you add new nets."""
    name = p.stem.lower()
    if ("mlp" in name) or ("perceptron" in name):
        return "mlp"
    return "mlp"  # default fallback


def build_model(arch_key: str) -> torch.nn.Module:
    if arch_key == "mlp":
        return MLPWide(0.35)
    raise ValueError(f"Unknown architecture key: {arch_key}")


# ------------------------- model discovery -------------------------
def discover_models(models_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Scan `models_dir` for *.pt and return a UI-friendly mapping:
      { "Mlp Model V001": {"path": ".../mlp_model_v001.pt", "arch": "mlp"}, ... }
    """
    out: Dict[str, Dict[str, str]] = {}
    models_dir = Path(models_dir)
    for f in sorted(models_dir.glob("*.pt")):
        ui_name = f.stem.replace("_", " ").title()
        out[ui_name] = {"path": str(f), "arch": detect_arch_from_filename(f)}
    if not out:
        raise FileNotFoundError(f"No .pt files found in {models_dir}")
    return out


# ------------------------- preprocess -------------------------
_to_tensor = transforms.ToTensor()

def preprocess_pil(pil: Image.Image, device: torch.device) -> Tuple[torch.Tensor, Image.Image]:
    """PIL -> ([1,1,28,28] tensor on `device`, native 28×28 PIL for display)."""
    pil = pil.convert("L").resize((28, 28), Image.BILINEAR)
    x = _to_tensor(pil)[0]  # [28, 28] in [0,1]

    # If background is light, invert to match training distribution
    if x.mean().item() > 0.5:
        x = 1.0 - x

    native_28 = Image.fromarray((x.cpu().numpy() * 255).astype(np.uint8))
    x = x.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,28,28]
    return x, native_28


# ------------------------- prediction (with tiny cache) -------------------------
class _Loaded:
    def __init__(self) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.name: Optional[str] = None
        self.arch: Optional[str] = None

_loaded = _Loaded()

def load_selected_model(all_models: Dict[str, Dict[str, str]],
                        model_name: str,
                        device: torch.device) -> torch.nn.Module:
    info = all_models[model_name]
    arch, path = info["arch"], info["path"]

    if (_loaded.model is None) or (_loaded.name != model_name) or (_loaded.arch != arch):
        m = build_model(arch).to(device)
        state = torch.load(path, map_location=device)
        m.load_state_dict(state)
        m.eval()
        _loaded.model, _loaded.name, _loaded.arch = m, model_name, arch

    return _loaded.model  # type: ignore[return-value]


@torch.no_grad()
def predict_from_pil(pil_img: Optional[Image.Image],
                     model_name: str,
                     all_models: Dict[str, Dict[str, str]],
                     device: torch.device):
    """
    Returns (probs: np.ndarray[10], top1_idx: int, native_28: PIL.Image)
    or (None, None, None) if `pil_img` is None.
    """
    if pil_img is None:
        return None, None, None

    model = load_selected_model(all_models, model_name, device)
    x, native_28 = preprocess_pil(pil_img, device)
    probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    top1 = int(np.argmax(probs))
    return probs, top1, native_28