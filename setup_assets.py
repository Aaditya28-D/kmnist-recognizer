# setup_assets.py
"""
Download required assets (test data + model weights) from Google Drive.
Run this once before using the app.

Usage:
    python setup_assets.py
"""

import gdown
from pathlib import Path

# Google Drive file IDs
FILES = {
    "data/kmnist-test-imgs.npz": "1bzZGMABGG95sKc7EBQmEs8rscsev86L8",
    "data/kmnist-test-labels.npz": "19Tdg0xJ11Ce1AYHtin2rYR7jlmWB7w8p",
    "models/mlp_model.pt": "1pyl1l7ribZ3lTgSdekWdU1LQJpUYI12c"
}

def ensure_assets():
    """Download missing assets into data/ and models/ folders."""
    for local_path, file_id in FILES.items():
        local_path = Path(local_path)
        if local_path.exists():
            print(f"✓ Found {local_path}, skipping download")
            continue

        # Make sure parent folder exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download from Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"↓ Downloading {local_path.name} ...")
        gdown.download(url, str(local_path), quiet=False)

if __name__ == "__main__":
    ensure_assets()