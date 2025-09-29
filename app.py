# app.py — KMNIST Gradio app (pretty labels + legend)

from pathlib import Path
import numpy as np
from PIL import Image
import gradio as gr

# Matplotlib for the prob. chart (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project imports (modularized)
from src.kmnist.infer import discover_models, predict_from_pil, pick_device
from src.kmnist.labels import KMNIST_CLASSES, KANA, PRON


# ------------------------- Paths -------------------------
def find_dir(dirname: str) -> Path:
    """Find a folder named `dirname` near CWD (./, ../, ../../)."""
    here = Path.cwd()
    for base in (here, here.parent, here.parent.parent):
        cand = base / dirname
        if cand.exists() and cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(f"Could not find '{dirname}' near {here}")

DATA_DIR   = find_dir("data")
MODELS_DIR = find_dir("models")

ALL_MODELS = discover_models(MODELS_DIR)
device     = pick_device()


# ------------------------- Pretty labels & helpers -------------------------
def pretty_label(romaji: str) -> str:
    """
    Format a class label like: o → お — “oh”
    Used in Top-1 text and in the Top-3 legend.
    """
    kana = KANA.get(romaji, "")
    pron = PRON.get(romaji, "")
    return f"`{romaji}` {kana} — {pron}"

def probs_to_fig(probs, classes):
    idx = np.argsort(-probs)[:10]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.bar(range(len(idx)), probs[idx])
    ax.set_xticks(range(len(idx)))
    # show romaji only; the Top-1 area shows kana + pron
    ax.set_xticklabels([classes[i] for i in idx])
    ax.set_ylim(0, 1)
    ax.set_ylabel("prob.")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig

# cached test set for the “Sample” tab
_KX = _KY = None
def load_kmnist_test():
    global _KX, _KY
    if _KX is not None:
        return _KX, _KY
    imgs = DATA_DIR / "kmnist-test-imgs.npz"
    labs = DATA_DIR / "kmnist-test-labels.npz"
    _KX = np.load(imgs)["arr_0"]   # (10000, 28, 28) uint8
    _KY = np.load(labs)["arr_0"]   # (10000,) int
    return _KX, _KY


# ------------------------- Predict wrappers -------------------------
def ui_predict_from_pil(pil_img: Image.Image, which_model: str):
    if pil_img is None:
        return "—", {}, None, None
    probs, top1, native_28 = predict_from_pil(pil_img, which_model, ALL_MODELS, device)

    romaji = KMNIST_CLASSES[top1]
    md = f"**Top-1:** {pretty_label(romaji)}  *(p={probs[top1]:.3f})*"

    # Top-3 dict keeps romaji keys (Gradio Label expects plain keys)
    top3 = {KMNIST_CLASSES[i]: float(probs[i]) for i in np.argsort(-probs)[:3]}
    fig  = probs_to_fig(probs, KMNIST_CLASSES)
    return md, top3, fig, native_28

def ui_predict_from_sketch(data, which_model: str):
    if data is None:
        return "—", {}, None, None

    # Gradio Sketchpad payload can be dict/array; unify to grayscale PIL
    arr = None
    if isinstance(data, dict):
        arr = data.get("image") or data.get("composite")
        if arr is None and data.get("layers"):
            layers = [np.asarray(l.get("image") if isinstance(l, dict) else l)
                      for l in data["layers"] if (l is not None)]
            if layers:
                arr = np.maximum.reduce(layers)
    else:
        arr = np.asarray(data)

    if arr is None:
        return "—", {}, None, None

    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)  # HxWxC -> gray
    arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    pil = Image.fromarray(arr).convert("L")

    return ui_predict_from_pil(pil, which_model)

def ui_predict_from_label(label_str: str, which_model: str):
    x_test, y_test = load_kmnist_test()
    if label_str not in KMNIST_CLASSES:
        return "—", {}, None, None, None
    label_idx = KMNIST_CLASSES.index(label_str)
    idxs = np.where(y_test == label_idx)[0]
    if len(idxs) == 0:
        return "—", {}, None, None, None

    pil28 = Image.fromarray(x_test[np.random.choice(idxs)]).convert("L")
    md, top3, fig, native_28 = ui_predict_from_pil(pil28, which_model)
    return md, top3, fig, native_28, pil28


# ------------------------- UI -------------------------
custom_css = """
.bold-head { font-weight: 700; font-size: 1.05rem; }
.kana-note { font-size: 0.92rem; opacity: 0.9; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("### ✨ KMNIST — Handwritten Kana (ひらがな) Recognizer")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("Pick a model, then upload, draw, or sample a character. "
                        "Images are shown at **true 28×28**; use the ⤢ button to zoom.")
            model_choice = gr.Dropdown(
                choices=list(ALL_MODELS.keys()),
                value=list(ALL_MODELS.keys())[0],
                label="Model",
            )

            with gr.Tab("Upload"):
                up = gr.Image(type="pil", image_mode="L", label="Upload")
                btn_up = gr.Button("Predict", variant="primary")

            with gr.Tab("Draw"):
                pad = gr.Sketchpad(label="Draw")
                btn_pad = gr.Button("Predict", variant="primary")

            with gr.Tab("Sample"):
                sample_label = gr.Dropdown(choices=KMNIST_CLASSES, value="ki", label="Pick label")
                btn_sample = gr.Button("Sample & predict", variant="secondary")

            with gr.Accordion("Kana legend (romaji → kana — pronunciation)", open=False):
                legend = "\n".join([f"- {pretty_label(lbl)}" for lbl in KMNIST_CLASSES])
                gr.Markdown(legend, elem_classes=["kana-note"])

        with gr.Column(scale=1):
            top1_out  = gr.Markdown("Top-1: —", elem_classes=["bold-head"])
            top3_out  = gr.Label(num_top_classes=3, label="Top-3 (romaji keys)")
            chart_out = gr.Plot(label="Probabilities")
            native28  = gr.Image(type="pil", label="Your 28×28 (native)",
                                  show_fullscreen_button=True, interactive=False)
            sample28  = gr.Image(type="pil", label="Sampled 28×28",
                                  show_fullscreen_button=True, interactive=False)

    btn_up.click(ui_predict_from_pil,     inputs=[up,  model_choice],
                 outputs=[top1_out, top3_out, chart_out, native28])
    btn_pad.click(ui_predict_from_sketch, inputs=[pad, model_choice],
                 outputs=[top1_out, top3_out, chart_out, native28])
    btn_sample.click(ui_predict_from_label, inputs=[sample_label, model_choice],
                 outputs=[top1_out, top3_out, chart_out, native28, sample28])

if __name__ == "__main__":
    demo.launch(share=True)