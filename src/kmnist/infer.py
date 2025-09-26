from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from .labels import KMNIST_CLASSES
from .models.mlp import MLPWide  # add more architectures here if needed

# ---------- device ----------
def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

# ---------- arch builder ----------
def detect_arch_from_filename(p: Path) -> str:
    name = p.stem.lower()
    if any(k in name for k in ["mlp", "perceptron"]): 
        return "mlp"
    return "mlp"

def build_model(arch_key: str):
    if arch_key == "mlp": 
        return MLPWide(0.35)
    raise ValueError(f"Unknown architecture key: {arch_key}")

# ---------- model discovery ----------
def discover_models(models_dir: Path) -> dict:
    out = {}
    for f in sorted(models_dir.glob("*.pt")):
        # friendly name for UI
        if f.name == "best.pt":
            ui_name = "Best Model"
        else:
            ui_name = f.stem.replace("_"," ").title()
        out[ui_name] = {"path": str(f), "arch": detect_arch_from_filename(f)}
    if not out:
        raise FileNotFoundError(f"No .pt files in {models_dir}")
    return out

# ---------- preprocess ----------
_to_tensor = transforms.ToTensor()

def preprocess_pil(pil: Image.Image, device: torch.device):
    pil = pil.convert("L").resize((28, 28), Image.BILINEAR)
    x = _to_tensor(pil)[0]  # [28,28] in [0,1]
    if x.mean().item() > 0.5:  # invert if background light
        x = 1.0 - x
    native_28 = Image.fromarray((x.cpu().numpy()*255).astype(np.uint8))  # for display
    x = x.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,28,28]
    return x, native_28

# ---------- prediction ----------
class Loaded:
    """Small cache to avoid rebuilding model on every call."""
    def __init__(self):
        self.model = None
        self.name  = None
        self.arch  = None

_loaded = Loaded()

def load_selected_model(all_models: dict, model_name: str, device: torch.device):
    info = all_models[model_name]
    arch = info["arch"]; path = info["path"]
    if _loaded.model is None or _loaded.name != model_name or _loaded.arch != arch:
        m = build_model(arch).to(device)
        state = torch.load(path, map_location=device)
        m.load_state_dict(state); m.eval()
        _loaded.model, _loaded.name, _loaded.arch = m, model_name, arch
    return _loaded.model

@torch.no_grad()
def predict_from_pil(pil_img: Image.Image, model_name: str, all_models: dict, device: torch.device):
    if pil_img is None: 
        return None, None, None
    model = load_selected_model(all_models, model_name, device)
    x, native_28 = preprocess_pil(pil_img, device)
    probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    top1 = int(np.argmax(probs))
    return probs, top1, native_28