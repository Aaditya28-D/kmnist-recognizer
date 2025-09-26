# ==== KMNIST Inference + Auto-Discovered Models (native 28×28, click to zoom) ====

import os
from pathlib import Path
import re
import numpy as np

# Headless backend for servers / Spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
import gradio as gr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ------------------------- Paths & discovery -------------------------
def find_dir(dirname: str) -> Path:
    """
    Find a directory named `dirname` relative to current working dir.
    Tries: ./dirname, ../dirname, ../../dirname
    """
    here = Path.cwd()
    for base in (here, here.parent, here.parent.parent):
        cand = base / dirname
        if cand.exists() and cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(f"Could not find '{dirname}' near {here}")

DATA_DIR   = find_dir("data")
MODELS_DIR = find_dir("models")


def _extract_version(p: Path) -> int | None:
    """
    Return integer version if filename contains _vNNN (e.g., v003 -> 3); else None.
    """
    m = re.search(r"_v(\d+)$", p.stem.lower())
    return int(m.group(1)) if m else None

def friendly_name_from_filename(p: Path) -> str:
    """
    - best.pt -> 'Best Model'
    - mlp_model_v003.pt -> 'Mlp Model (v3)'
    - multilayer_perceptron.pt -> 'Multilayer Perceptron'
    """
    if p.name.lower() == "best.pt":
        return "Best Model"
    stem = p.stem.replace("_", " ").title()
    v = _extract_version(p)
    return f"{stem} (v{v})" if v is not None else stem

def detect_arch_from_filename(p: Path) -> str:
    """
    Decide which architecture to build based on filename.
    Extend with your own rules (cnn, resnet, etc).
    """
    name = p.stem.lower()
    if ("mlp" in name) or ("perceptron" in name):
        return "mlp"
    return "mlp"  # default


def discover_models(models_dir: Path) -> dict:
    """
    Scan models_dir for *.pt files and return an ordered dict-like mapping:
    {ui_name: {"path": str, "arch": "mlp" | ...}}
    Priority:
      1) best.pt first (if present)
      2) Then by version number (descending) for *_vNNN.pt
      3) Then alphabetical
    """
    files = sorted(models_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(
            f"No .pt files found in {models_dir}. "
            "Place your trained weights (e.g. 'mlp_model_v001.pt' or 'best.pt') there."
        )

    def sort_key(p: Path):
        # best.pt should come first
        if p.name.lower() == "best.pt":
            return (0, 0, "")  # top priority
        v = _extract_version(p)
        # Put versioned models next, sorted by version DESC
        if v is not None:
            return (1, -v, p.stem.lower())
        # Non-versioned fall to the bottom, alphabetically
        return (2, 0, p.stem.lower())

    files_sorted = sorted(files, key=sort_key)

    out = {}
    for f in files_sorted:
        out[friendly_name_from_filename(f)] = {
            "path": str(f),
            "arch": detect_arch_from_filename(f),
        }
    return out

ALL_MODELS = discover_models(MODELS_DIR)


# ------------------------- Device -------------------------
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = pick_device()


# ------------------------- Architectures -------------------------
class MLP_Wide(nn.Module):
    def __init__(self, p=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(p),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(p),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(p),
            nn.Linear(128, 10)
        )
    def forward(self, x):  # x: [B, 1, 28, 28]
        return self.net(x)


def build_model(arch_key: str) -> nn.Module:
    if arch_key == "mlp":
        return MLP_Wide(0.35)
    raise ValueError(f"Unknown architecture key: {arch_key}")


# cache the currently loaded model so we don't reload on every click
_current = {"model": None, "arch": None, "name": None}

def load_selected_model(model_name: str) -> nn.Module:
    info = ALL_MODELS[model_name]
    arch = info["arch"]
    path = info["path"]

    if (
        _current["model"] is None
        or _current["arch"] != arch
        or _current["name"] != model_name
    ):
        m = build_model(arch).to(device)
        state = torch.load(path, map_location=device)
        m.load_state_dict(state)
        m.eval()
        _current.update({"model": m, "arch": arch, "name": model_name})
    return _current["model"]


# ------------------------- Labels -------------------------
KMNIST_CLASSES = ["o","ki","su","tsu","na","ha","ma","ya","re","wo"]
KANA = {"o":"お","ki":"き","su":"す","tsu":"つ","na":"な","ha":"は","ma":"ま","ya":"や","re":"れ","wo":"を"}
PRON = {"o":"“oh”","ki":"“kee”","su":"“soo”","tsu":"“tsoo”","na":"“nah”","ha":"“hah”","ma":"“mah”","ya":"“yah”","re":"“reh”","wo":"“oh/wo”"}


# ------------------------- Test set (for Sample tab) -------------------------
_KX = _KY = None
def load_kmnist_test():
    global _KX, _KY
    if _KX is not None:
        return _KX, _KY
    imgs = DATA_DIR / "kmnist-test-imgs.npz"
    labs = DATA_DIR / "kmnist-test-labels.npz"
    if not imgs.exists() or not labs.exists():
        raise FileNotFoundError(f"KMNIST test .npz not found in {DATA_DIR}")
    _KX = np.load(imgs)["arr_0"]  # (10000, 28, 28)
    _KY = np.load(labs)["arr_0"]  # (10000,)
    return _KX, _KY


# ------------------------- Helpers -------------------------
to_tensor = transforms.ToTensor()

def preprocess_pil(pil: Image.Image):
    """PIL -> ([1,1,28,28] tensor on device, native 28×28 PIL)."""
    pil = pil.convert("L").resize((28, 28), Image.BILINEAR)
    x = to_tensor(pil)[0]  # [28,28] in [0,1]
    if x.mean().item() > 0.5:  # invert if background light
        x = 1.0 - x
    native_28 = Image.fromarray((x.cpu().numpy() * 255).astype(np.uint8))
    return x.unsqueeze(0).unsqueeze(0).to(device), native_28


def probs_to_fig(probs, classes):
    idx = np.argsort(-probs)[:10]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.bar(range(len(idx)), probs[idx])
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([classes[i] for i in idx])
    ax.set_ylim(0, 1)
    ax.set_ylabel("prob.")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def small_ref_grid(label_idx: int, grid=2):
    """2×2 grid at native resolution (56×56)."""
    try:
        x_test, y_test = load_kmnist_test()
    except Exception:
        return Image.fromarray(np.zeros((56, 56), np.uint8))
    idxs = np.where(y_test == label_idx)[0]
    if len(idxs) == 0:
        return Image.fromarray(np.zeros((56, 56), np.uint8))
    rng = np.random.default_rng(12345 + label_idx)
    picks = rng.choice(idxs, size=min(grid * grid, len(idxs)), replace=False)
    canvas = Image.new("L", (28 * grid, 28 * grid), 0)
    r = c = 0
    for idx in picks:
        img = 255 - x_test[idx].astype(np.uint8)  # white-on-black like model input
        canvas.paste(Image.fromarray(img), (c * 28, r * 28))
        c += 1
        if c == grid:
            c = 0
            r += 1
            if r == grid:
                break
    return canvas


# ------------------------- Predictors -------------------------
@torch.no_grad()
def predict_from_pil(pil_img: Image.Image, which_model: str):
    if pil_img is None:
        return "—", {}, None, None, None
    model = load_selected_model(which_model)
    x, native_28 = preprocess_pil(pil_img)
    probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    top1 = int(np.argmax(probs))
    romaji = KMNIST_CLASSES[top1]
    md = f"**Top-1:** `{romaji}` {KANA[romaji]} *(p={probs[top1]:.3f}; {PRON[romaji]})*"
    top3 = {KMNIST_CLASSES[i]: float(probs[i]) for i in np.argsort(-probs)[:3]}
    fig  = probs_to_fig(probs, KMNIST_CLASSES)
    ref  = small_ref_grid(top1, grid=2)
    return md, top3, fig, native_28, ref


@torch.no_grad()
def predict_from_sketch(data, which_model: str):
    if data is None:
        return "—", {}, None, None, None

    # Gradio Sketchpad payload is dict or array; handle both
    arr = None
    if isinstance(data, dict):
        if data.get("image") is not None:
            arr = data["image"]
        elif data.get("composite") is not None:
            arr = data["composite"]
        elif data.get("layers") is not None:
            imgs = []
            for layer in data["layers"]:
                li = layer.get("image") if isinstance(layer, dict) else layer
                if li is not None:
                    imgs.append(np.asarray(li))
            if imgs:
                arr = np.maximum.reduce(imgs)
    else:
        arr = np.asarray(data)

    if arr is None:
        return "—", {}, None, None, None

    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)  # HxWxC -> gray
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    pil = Image.fromarray(arr).convert("L")
    return predict_from_pil(pil, which_model)


@torch.no_grad()
def predict_from_label(label_str: str, which_model: str):
    try:
        x_test, y_test = load_kmnist_test()
    except Exception as e:
        return f"— (test set not found: {e})", {}, None, None, None, None
    if label_str not in KMNIST_CLASSES:
        return "—", {}, None, None, None, None

    label_idx = KMNIST_CLASSES.index(label_str)
    idxs = np.where(y_test == label_idx)[0]
    if len(idxs) == 0:
        return "—", {}, None, None, None, None

    pil28 = Image.fromarray(x_test[np.random.choice(idxs)]).convert("L")
    pil28_inv = Image.fromarray(255 - np.array(pil28))  # native 28×28, white-on-black
    md, top3, fig, native_28, ref = predict_from_pil(pil28, which_model)
    return md, top3, fig, native_28, ref, pil28_inv


# ------------------------- UI -------------------------
with gr.Blocks() as demo:
    gr.Markdown("## KMNIST — Handwritten Kana (ひらがな) Recognizer")
    gr.Markdown(
        "Upload, draw, or sample a character. Images are shown at **true 28×28**; "
        "use the ⤢ button on any image to zoom. Choose a model below — "
        "the list is auto-discovered from your `models/` folder."
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                choices=list(ALL_MODELS.keys()),
                value=list(ALL_MODELS.keys())[0],
                label="Select Model",
                multiselect=False
            )

            with gr.Tab("Upload"):
                up = gr.Image(type="pil", image_mode="L", label="Upload")
                btn_up = gr.Button("Predict from upload")

            with gr.Tab("Draw"):
                pad = gr.Sketchpad(label="Draw")
                btn_pad = gr.Button("Predict from drawing")

            with gr.Tab("Sample"):
                sample_label = gr.Dropdown(choices=KMNIST_CLASSES, value="ki", label="Pick class")
                btn_sample = gr.Button("Sample & predict")

        with gr.Column(scale=1):
            top1_out  = gr.Markdown("Top-1: —")
            top3_out  = gr.Label(num_top_classes=3, label="Top-3")
            chart_out = gr.Plot(label="Probabilities")
            your28_out = gr.Image(type="pil", label="Your 28×28 (native)",
                                  show_fullscreen_button=True, interactive=False)
            ref_out    = gr.Image(type="pil", label="Reference (2×2, 56×56 native)",
                                  show_fullscreen_button=True, interactive=False)
            sample_out = gr.Image(type="pil", label="Sampled test image (28×28 native)",
                                  show_fullscreen_button=True, interactive=False)

    btn_up.click(predict_from_pil,     inputs=[up,  model_choice],
                 outputs=[top1_out, top3_out, chart_out, your28_out, ref_out])
    btn_pad.click(predict_from_sketch, inputs=[pad, model_choice],
                 outputs=[top1_out, top3_out, chart_out, your28_out, ref_out])
    btn_sample.click(predict_from_label, inputs=[sample_label, model_choice],
                 outputs=[top1_out, top3_out, chart_out, your28_out, ref_out, sample_out])


if __name__ == "__main__":
    demo.launch(share=True)