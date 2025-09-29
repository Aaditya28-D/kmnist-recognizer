# setup_assets.py
"""
Download required assets (test data + ALL model weights listed in models_index.json)
Run this once after cloning the repo.

Usage:
    python setup_assets.py
"""

from pathlib import Path
import json
import gdown

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
INDEX_PATH = ROOT / "models_index.json"

# ---- Static data files (Google Drive IDs) ----
DATA_FILES = {
    "data/kmnist-test-imgs.npz":   "1bzZGMABGG95sKc7EBQmEs8rscsev86L8",
    "data/kmnist-test-labels.npz": "19Tdg0xJ11Ce1AYHtin2rYR7jlmWB7w8p",
}

def gdown_id(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"

def ensure(path: Path, file_id: str):
    """Download file if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"✓ Found {path}, skipping")
        return
    url = gdown_id(file_id)
    print(f"↓ Downloading {path} ...")
    gdown.download(url, str(path), quiet=False)

def load_models_index() -> list[dict]:
    """Return a list of {'filename':..., 'file_id':...} from models_index.json (if present)."""
    if not INDEX_PATH.exists():
        print(f"⚠️  {INDEX_PATH} not found — skipping model downloads.")
        return []
    try:
        idx = json.loads(INDEX_PATH.read_text())
        return list(idx.get("models", []))
    except Exception as e:
        print(f"⚠️  Could not read {INDEX_PATH}: {e}")
        return []

def main():
    # Data (NPZs)
    for rel, fid in DATA_FILES.items():
        ensure(ROOT / rel, fid)

    # Models
    MODELS_DIR.mkdir(exist_ok=True)
    entries = load_models_index()
    if not entries:
        return

    for entry in entries:
        fname = entry.get("filename")
        fid   = entry.get("file_id")
        if not fname or not fid:
            continue
        ensure(MODELS_DIR / fname, fid)

if __name__ == "__main__":
    main()