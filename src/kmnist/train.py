# src/kmnist/train.py
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    val_loader: Optional[DataLoader] = None,
    out_path: Optional[Path] = None,
    # --- new early-stopping options ---
    early_stopping: bool = False,
    patience: int = 5,
    min_delta: float = 1e-3,
    save_best_only: bool = False,
) -> Dict[str, float]:
    """
    Train `model`. If `val_loader` is provided, tracks val loss/acc and supports
    early stopping and saving only the best checkpoint.
    """
    device = pick_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
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

            # --- checkpointing ---
            improved = (best_val_loss - va_loss) > min_delta
            if improved:
                best_val_loss = va_loss
                epochs_no_improve = 0
                if save_best_only and out_path is not None:
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    torch.save(best_state, out_path)
            else:
                epochs_no_improve += 1

            # --- early stopping ---
            if early_stopping and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        else:
            print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} loss {tr_loss:.3f}")

    # if not saving-best-only during training, save final weights if requested
    if out_path is not None and not save_best_only:
        torch.save(model.state_dict(), out_path)
        print(f"Saved weights -> {out_path}")

    return {
        "train_loss": tr_loss,
        "train_acc": tr_acc,
        "val_loss": best_val_loss if val_loader is not None else float("nan"),
    }