# src/kmnist/train.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Tuple
import re
import shutil

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


# ------------------------- device -------------------------
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


# ------------------------- naming helpers -------------------------
_VERSION_RX = re.compile(r"_v(\d{3})", re.IGNORECASE)
_ACC_RX     = re.compile(r"_acc([0-9]+\.[0-9]{3})", re.IGNORECASE)

def _as_models_dir_and_prefix(out_path: Optional[Path]) -> Tuple[Path, str]:
    """
    Accepts either a file path (e.g. models/mlp_model.pt) or a folder (models/).
    Returns (models_dir, base_prefix).
    """
    if out_path is None:
        # default folder + prefix
        return Path("models"), "mlp_model"
    p = Path(out_path)
    if p.suffix == ".pt":
        return p.parent if p.parent != Path("") else Path("models"), p.stem
    # if it's a directory or no suffix, use it as folder with default prefix
    return p, "mlp_model"

def _next_version(models_dir: Path, prefix: str) -> int:
    max_v = 0
    for f in models_dir.glob(f"{prefix}_v*.pt"):
        m = _VERSION_RX.search(f.stem)
        if m:
            try:
                max_v = max(max_v, int(m.group(1)))
            except ValueError:
                pass
    return max_v + 1

def _current_best_acc(models_dir: Path, prefix: str) -> float:
    """
    Finds best accuracy from existing versioned files. Falls back to reading best.pt name if present.
    """
    best = -1.0
    # scan versioned files
    for f in models_dir.glob(f"{prefix}_v*_acc*.pt"):
        m = _ACC_RX.search(f.stem)
        if m:
            try:
                best = max(best, float(m.group(1)))
            except ValueError:
                pass
    # optional hint from best alias like best_acc0.912.pt
    for f in models_dir.glob("best_acc*.pt"):
        m = _ACC_RX.search(f.stem)
        if m:
            try:
                best = max(best, float(m.group(1)))
            except ValueError:
                pass
    return best

def _save_versioned_and_maybe_update_best(
    state_dict: Dict[str, torch.Tensor],
    models_dir: Path,
    prefix: str,
    achieved_acc: float,
) -> Path:
    """
    Saves a new versioned checkpoint and updates 'best.pt' (+ best_acc*.pt alias)
    if it beats the previous best.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    version = _next_version(models_dir, prefix)
    acc_tag = f"acc{achieved_acc:.3f}"
    final_path = models_dir / f"{prefix}_v{version:03d}_{acc_tag}.pt"
    torch.save(state_dict, final_path)

    prev_best = _current_best_acc(models_dir, prefix)
    if achieved_acc > prev_best:
        # Update easy alias
        alias_best = models_dir / "best.pt"
        torch.save(state_dict, alias_best)
        # Also keep an explicit “best_acc*.pt” snapshot for clarity
        alias_named = models_dir / f"best_{acc_tag}.pt"
        if alias_named.exists():
            alias_named.unlink()
        shutil.copyfile(final_path, alias_named)
        print(f"[ckpt] New BEST → {alias_best}  (prev={prev_best:.3f} → new={achieved_acc:.3f})")
    else:
        print(f"[ckpt] Saved {final_path.name} (best so far remains {prev_best:.3f})")

    return final_path


# ------------------------- training -------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    val_loader: Optional[DataLoader] = None,
    out_path: Optional[Path] = None,
    # early stopping
    early_stopping: bool = False,
    patience: int = 5,
    min_delta: float = 1e-3,
    save_best_only: bool = False,
) -> Dict[str, float]:
    """
    Train `model`. If `val_loader` is provided, tracks val loss/acc and supports
    early stopping and saving only the best checkpoint.

    Checkpoint behavior:
      - Always saves a versioned file like:  {prefix}_vNNN_accX.XXX.pt
      - Also updates 'best.pt' (+ 'best_accX.XXX.pt') only if the new model is better.
    """
    device = pick_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    models_dir, prefix = _as_models_dir_and_prefix(Path(out_path) if out_path else None)

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_acc = -1.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_loss_sum, tr_acc_sum, tr_count = 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            tr_loss_sum += loss.item() * bs
            tr_acc_sum  += _accuracy(logits, yb) * bs
            tr_count    += bs

        tr_loss = tr_loss_sum / tr_count
        tr_acc  = tr_acc_sum  / tr_count

        # ---- validate (optional) ----
        if val_loader is not None:
            model.eval()
            va_loss_sum, va_acc_sum, va_count = 0.0, 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    bs = yb.size(0)
                    va_loss_sum += loss.item() * bs
                    va_acc_sum  += _accuracy(logits, yb) * bs
                    va_count    += bs

            va_loss = va_loss_sum / va_count
            va_acc  = va_acc_sum  / va_count

            print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} loss {tr_loss:.3f} "
                  f"| val acc {va_acc:.3f} loss {va_loss:.3f}")

            # --- checkpointing (track best during training) ---
            improved = (best_val_loss - va_loss) > min_delta
            if improved or (va_loss < best_val_loss and best_val_loss == float("inf")):
                best_val_loss = va_loss
                best_val_acc  = va_acc
                epochs_no_improve = 0
                if save_best_only:
                    # keep a CPU copy so it can be saved later regardless of device
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            # --- early stopping ---
            if early_stopping and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        else:
            print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} loss {tr_loss:.3f}")

    # ------------------------- final save -------------------------
    if save_best_only and val_loader is not None and best_state is not None:
        state_to_save = best_state
        achieved_acc  = best_val_acc
    else:
        state_to_save = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        achieved_acc  = tr_acc if val_loader is None else best_val_acc

    if achieved_acc is None or achieved_acc < 0:
        # Fallback if no metric recorded (e.g., no val loader)
        achieved_acc = tr_acc

    saved_path = _save_versioned_and_maybe_update_best(
        state_dict=state_to_save,
        models_dir=models_dir,
        prefix=prefix,
        achieved_acc=float(achieved_acc),
    )

    return {
        "train_loss": float(tr_loss),
        "train_acc":  float(tr_acc),
        "val_loss":   float(best_val_loss) if val_loader is not None else float("nan"),
        "model_path": str(saved_path),
        "best_val_acc": float(best_val_acc),
    }
