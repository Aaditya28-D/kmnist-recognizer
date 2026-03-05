# src/kmnist/assets.py
"""
Asset bootstrap for lazy users.

Ensures required assets exist:
- data/kmnist-test-imgs.npz
- data/kmnist-test-labels.npz
- all model files listed in models_index.json

If anything is missing, runs setup_assets.py automatically.
"""

from pathlib import Path
import subprocess
import sys
import json


def project_root() -> Path:
    # .../src/kmnist/assets.py -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def _required_data_files(root: Path) -> list[Path]:
    return [
        root / "data" / "kmnist-test-imgs.npz",
        root / "data" / "kmnist-test-labels.npz",
    ]


def _required_model_files(root: Path) -> list[Path]:
    index_path = root / "models_index.json"
    if not index_path.exists():
        # If index is missing, we can't know required models.
        # We'll just rely on setup_assets.py when models folder is empty/missing.
        return []

    try:
        idx = json.loads(index_path.read_text())
        entries = idx.get("models", [])
        out: list[Path] = []
        for e in entries:
            fname = e.get("filename")
            if fname:
                out.append(root / "models" / fname)
        return out
    except Exception:
        # Corrupt/invalid json -> fall back to folder existence check
        return []


def _missing(paths: list[Path]) -> bool:
    return any(not p.exists() for p in paths)


def ensure_assets():
    """
    If assets are missing, run setup_assets.py.
    This makes `python3 app.py` work from a fresh clone with no manual steps.
    """
    root = project_root()

    data_dir = root / "data"
    models_dir = root / "models"
    setup_script = root / "setup_assets.py"

    # Basic folder existence
    data_ok = data_dir.exists() and data_dir.is_dir()
    models_ok = models_dir.exists() and models_dir.is_dir()

    # Specific required files (stronger, lazy-user proof)
    req_data = _required_data_files(root)
    req_models = _required_model_files(root)

    need_setup = (
        (not data_ok)
        or (not models_ok)
        or _missing(req_data)
        or (req_models != [] and _missing(req_models))
        or (models_ok and not any(models_dir.iterdir()))  # empty models folder
    )

    if not need_setup:
        return

    if not setup_script.exists():
        raise FileNotFoundError(f"Missing setup script: {setup_script}")

    print("Required assets missing. Running setup_assets.py ...")

    # Run using same interpreter; set cwd to repo root for reliability
    subprocess.check_call([sys.executable, str(setup_script)], cwd=str(root))