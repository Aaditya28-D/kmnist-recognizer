# Changelog

All notable changes to this project will be documented in this file.  
This project follows [Semantic Versioning](https://semver.org/).

---

## [v1.0] - 2025-09-29
### Added
- Initial **stable release** of KMNIST Recognizer 🎉
- `app.py`: Lean Gradio app with Japanese Kana labels (Romaji + Kana + Pronunciation).
- `setup_assets.py`: Downloads required test assets from Google Drive.
- `upload_models.py`: Automated model uploads to Google Drive with `models_index.json` sync.
- Modular `src/kmnist/` structure:
  - `infer.py`: Model discovery & inference utilities.
  - `train.py`: Model training with versioned checkpoints.
  - `models/mlp.py`: MLPWide architecture.
  - `labels.py`: Kana labels + pronunciation.
  - `data_utils.py`: Dataset loaders and utilities.

### Changed
- Cleaned project structure for easier navigation.
- Updated `.gitignore`:
  - Keeps code in `src/models/`
  - Ignores only `.pt` model weight files in `/models`.

### Notes
- Models are **stored in Google Drive**, not GitHub.
- This release is tagged as `v1.0` on GitHub.

---