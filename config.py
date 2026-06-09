# config.py
MODEL_PATH = 'models/model.keras'
RETINAL_MODEL_PATH = 'models/retinal.h5'
ALLOW_REMOTE_MODEL_DOWNLOAD = False

# Public HTTPS URLs where the model files are hosted.
# GitHub Releases direct download URLs
MODEL_DOWNLOAD_URLS = {
    'model.keras': 'https://github.com/RodrickGulu/glaucoma_detection/releases/download/v1.0-glaucoma-models/model.keras',
    'retinal.h5': 'https://github.com/RodrickGulu/glaucoma_detection/releases/download/v1.0-glaucoma-models/retinal.h5'
}

# Example:
# MODEL_DOWNLOAD_URLS['model.keras'] = 'https://your-storage.example.com/model.keras'
# MODEL_DOWNLOAD_URLS['retinal.h5'] = 'https://your-storage.example.com/retinal.h5'