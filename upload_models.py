# upload_models.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


# Paths anchored to repo root (this script’s folder)
ROOT         = Path(__file__).resolve().parent
MODELS_DIR   = ROOT / "models"
INDEX_PATH   = ROOT / "models_index.json"
SECRETS_JSON = ROOT / "client_secrets.json"
TOKEN_JSON   = ROOT / "token.json"


def load_or_init_index(folder_id_arg: str | None) -> Dict[str, Any]:
    """
    Load models_index.json if present; otherwise initialize a new one.
    Drive folder id resolution priority:
      1) --folder-id CLI arg
      2) DRIVE_FOLDER_ID env var
      3) existing value in models_index.json (if present)
    """
    idx: Dict[str, Any] = {"drive_folder_name": "models", "drive_folder_id": "", "models": []}

    if INDEX_PATH.exists():
        try:
            idx = json.loads(INDEX_PATH.read_text())
        except Exception as e:
            raise RuntimeError(f"Failed to read {INDEX_PATH}: {e}")

    folder_id = (
        folder_id_arg
        or os.getenv("DRIVE_FOLDER_ID")
        or idx.get("drive_folder_id")
        or ""
    )
    if not folder_id:
        raise ValueError(
            "No Drive folder id provided.\n"
            "Pass --folder-id, or set env DRIVE_FOLDER_ID, or put it under "
            f"'drive_folder_id' in {INDEX_PATH.name}."
        )

    idx["drive_folder_id"] = folder_id
    idx.setdefault("models", [])
    return idx


def save_index(idx: Dict[str, Any]) -> None:
    INDEX_PATH.write_text(json.dumps(idx, indent=2))
    print(f"✓ Wrote {INDEX_PATH.relative_to(ROOT)}")


def get_drive() -> GoogleDrive:
    if not SECRETS_JSON.exists():
        raise FileNotFoundError(
            f"Missing {SECRETS_JSON.name} in repo root. "
            "Download OAuth client credentials (Desktop app) from Google Cloud and save it there."
        )

    gauth = GoogleAuth()
    gauth.LoadClientConfigFile(str(SECRETS_JSON))

    if TOKEN_JSON.exists():
        gauth.LoadCredentialsFile(str(TOKEN_JSON))
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
        gauth.SaveCredentialsFile(str(TOKEN_JSON))
    elif gauth.access_token_expired:
        gauth.Refresh()
        gauth.SaveCredentialsFile(str(TOKEN_JSON))
    else:
        gauth.Authorize()

    return GoogleDrive(gauth)


def make_file_public(drive: GoogleDrive, file_id: str) -> None:
    """
    Set Drive file permission to Anyone-with-link (viewer).
    Uses the underlying Drive v3 API for reliability.
    """
    # allowFileDiscovery=False means "Anyone with the link" (not searchable)
    body = {"type": "anyone", "role": "reader", "allowFileDiscovery": False}
    drive.auth.service.permissions().create(fileId=file_id, body=body).execute()


def upload_new_models(idx: Dict[str, Any]) -> bool:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    known: Dict[str, str] = {m["filename"]: m["file_id"] for m in idx.get("models", [])}
    candidates: List[Path] = sorted(p for p in MODELS_DIR.glob("*.pt") if p.is_file())

    if not candidates:
        print(f"(no *.pt files under {MODELS_DIR.relative_to(ROOT)})")
        return False

    drive = get_drive()
    folder_id = idx["drive_folder_id"]

    uploaded_any = False
    for f in candidates:
        if f.name in known:
            print(f"= Skipping (already indexed): {f.name}")
            continue

        print(f"↑ Uploading: {f.name}")
        gfile = drive.CreateFile({"title": f.name, "parents": [{"id": folder_id}]})
        gfile.SetContentFile(str(f))
        gfile.Upload()
        file_id = gfile["id"]
        print(f"  → Drive file id: {file_id}")

        # NEW: make public so gdown can fetch it without manual steps
        try:
            make_file_public(drive, file_id)
            print("  → Sharing: Anyone with the link (viewer)")
        except Exception as e:
            print(f"  ! Failed to set public permission: {e}")
            print("    You can set sharing manually in Drive if needed.")

        idx.setdefault("models", []).append({"filename": f.name, "file_id": file_id})
        uploaded_any = True

    return uploaded_any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload new .pt models to Google Drive, make them public, and update models_index.json"
    )
    parser.add_argument("--folder-id", help="Google Drive folder id (overrides file/env)", default=None)
    args = parser.parse_args()

    idx = load_or_init_index(args.folder_id)
    changed = upload_new_models(idx)
    if changed:
        save_index(idx)
    else:
        print("Nothing to do. Index is up to date.")


if __name__ == "__main__":
    main()