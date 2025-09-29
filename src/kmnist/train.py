# src/kmnist/train.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List
import re

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim


# --------- device ----------
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


# --------- versioned filename helper ----------
def _next_versioned_path(out_dir: Path, prefix: str) -> Path:
    """
    Find the next 'prefix_vNNN.pt' file name inside out_dir.
    Example: mlp_model_v001.pt, mlp_model_v002.pt, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing: List[Path] = list(out_dir.glob(f"{prefix}_v*.pt"))
    if not existing:
        return out_dir / f"{prefix}_v001.pt"

    nums = []
    for f in existing:
        m = re.search(r"_v(\d+)\.pt$", f.name)
        if m:
            nums.append(int(m.group(1)))
    n = max(nums) + 1 if nums else 1
    return out_dir / f"{prefix}_v{n:03d}.pt"


# --------- training ----------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    val_loader: Optional[DataLoader] = None,
    out_dir: Path | str = "models",
    prefix: str = "mlp_model",
) -> Dict[str, float]:
    """
    Minimal training loop.
    - Prints train (and optional val) metrics each epoch.
    - Saves ONE new file at the end: '<prefix>_vNNN.pt' in out_dir.
    - No 'best model' tracking, no early stopping.
    """
    out_dir = Path(out_dir)
    device = pick_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_acc_sum = 0.0
        count = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            tr_loss_sum += loss.item() * bs
            tr_acc_sum += _accuracy(logits, yb) * bs
            count += bs

        tr_loss = tr_loss_sum / max(count, 1)
        tr_acc = tr_acc_sum / max(count, 1)

        # optional validation
        if val_loader is not None:
            model.eval()
            va_loss_sum = 0.0
            va_acc_sum = 0.0
            va_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    bs = yb.size(0)
                    va_loss_sum += loss.item() * bs
                    va_acc_sum += _accuracy(logits, yb) * bs
                    va_count += bs

            va_loss = va_loss_sum / max(va_count, 1)
            va_acc = va_acc_sum / max(va_count, 1)
            print(
                f"Epoch {epoch:02d} | train acc {tr_acc:.3f} loss {tr_loss:.3f} "
                f"| val acc {va_acc:.3f} loss {va_loss:.3f}"
            )
        else:
            print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} loss {tr_loss:.3f}")

    # save one new versioned file
    out_path = _next_versioned_path(out_dir, prefix)
    torch.save(model.state_dict(), out_path)
    print(f"Saved weights -> {out_path}")

    return {"train_loss": tr_loss, "train_acc": tr_acc}