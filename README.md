# ✨ KMNIST Recognizer (Gradio)

A simple KMNIST handwritten kana recognizer with a Gradio UI.

✅ **Lazy-user friendly**: On first run, the app automatically downloads required assets (test data + models) from Google Drive.

---

## 🔗 Live Demo
Run on Hugging Face Spaces (no setup needed).

---

## 🚀 Run Locally

### 1) Clone the repository
```bash
git clone https://github.com/Aaditya28-D/kmnist-recognizer.git
cd kmnist-recognizer
```

### 2) Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run the app
```bash
python3 app.py
```

On the **first run**, the app automatically:

- downloads KMNIST test data into `data/`
- downloads trained models into `models/`
- opens the UI in your browser

If the browser doesn’t open, go to:

http://127.0.0.1:7860

---

## 🌍 Optional: Enable a public share link

By default, the app runs locally (faster).  
If you want a temporary public Gradio link, run:

```bash
GRADIO_SHARE=1 python3 app.py
```

---

## 🧠 Project Structure (important files)

- `app.py` — Gradio UI entrypoint  
- `src/kmnist/infer.py` — inference pipeline  
- `src/kmnist/models/mlp.py` — model architecture  
- `src/kmnist/assets.py` — auto-download assets if missing  
- `src/kmnist/data.py` — loads test dataset  
- `setup_assets.py` — downloads assets from Google Drive  
- `models_index.json` — model registry (filenames + Drive IDs)

---

## 🧪 Training

Training notebooks are available in:

notebooks/
