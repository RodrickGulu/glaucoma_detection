# glaucoma_detection

## Remote Model Hosting
This app can download the trained models at startup rather than storing them in the repository.

1. Upload `models/model.keras` and `models/retinal.h5` to a stable public HTTPS host.
2. Set the URLs in `config.py` under `MODEL_DOWNLOAD_URLS`.
3. Keep the model files out of source control; `.gitignore` now excludes them.
4. On startup, the app will download any missing model file and save it locally.

Supported storage options:
- GitHub Releases or GitHub Pages
- AWS S3 / DigitalOcean Spaces
- Google Cloud Storage
- Azure Blob Storage
- any HTTPS file host with direct download links
