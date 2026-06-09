import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from PIL import Image
from datetime import datetime
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from model import load_modell, predict
from retinal import predict_image
from config import MODEL_PATH, RETINAL_MODEL_PATH, MODEL_DOWNLOAD_URLS, ALLOW_REMOTE_MODEL_DOWNLOAD

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Define a directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
MASK_FOLDER = 'masks'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MASK_FOLDER'], exist_ok=True)


def download_remote_file(url, destination_path, chunk_size=8192):
    destination_dir = os.path.dirname(destination_path)
    os.makedirs(destination_dir, exist_ok=True)

    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['HEAD', 'GET', 'OPTIONS']
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    print(f"Downloading model from: {url}")
    try:
        with session.get(url, stream=True, timeout=300, headers=headers) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0) or 0)
            downloaded = 0
            with open(destination_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"Progress: {percent:.1f}%", end='\r')

        print(f"\nSuccessfully downloaded to: {destination_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading from {url}: {str(e)}")
        if os.path.exists(destination_path):
            os.remove(destination_path)
        raise


def ensure_model_files():
    if not ALLOW_REMOTE_MODEL_DOWNLOAD:
        print('Remote model download is disabled for this deployment. Using lightweight fallback mode.')
        return

    remote_mapping = {
        MODEL_PATH: MODEL_DOWNLOAD_URLS.get(os.path.basename(MODEL_PATH)),
        RETINAL_MODEL_PATH: MODEL_DOWNLOAD_URLS.get(os.path.basename(RETINAL_MODEL_PATH))
    }

    for local_path, remote_url in remote_mapping.items():
        if os.path.exists(local_path):
            print(f"Model found locally: {local_path}")
            continue
        
        if not remote_url:
            print(f"Warning: No remote URL configured for {local_path}")
            raise RuntimeError(f'Missing remote URL for {local_path}. Set MODEL_DOWNLOAD_URLS in config.py.')
        
        try:
            print(f"Downloading model from remote URL for: {local_path}")
            download_remote_file(remote_url, local_path)
        except Exception as e:
            raise RuntimeError(f'Failed to download model from {remote_url}: {str(e)}')


# Global variables for lazy loading
_model = None
_model1 = None


def get_glaucoma_model():
    """Load the glaucoma classifier only when needed."""
    global _model
    if _model is None:
        try:
            ensure_model_files()
            if not os.path.exists(MODEL_PATH):
                print('Glaucoma model file is unavailable; using fallback prediction mode.')
                return None
            _model = load_modell(MODEL_PATH)
            print("Glaucoma model loaded successfully")
        except Exception as e:
            print(f"Error loading glaucoma model: {str(e)}")
            return None
    return _model


def get_retinal_model():
    """Load the retinal validation model only when needed."""
    global _model1
    if _model1 is None:
        try:
            ensure_model_files()
            if not os.path.exists(RETINAL_MODEL_PATH):
                print('Retinal model file is unavailable; using fallback validation mode.')
                return None
            _model1 = load_model(RETINAL_MODEL_PATH)
            print("Retinal model loaded successfully")
        except Exception as e:
            print(f"Error loading retinal model: {str(e)}")
            return None
    return _model1


def get_models():
    """Compatibility wrapper for callers that still need both models."""
    return get_glaucoma_model(), get_retinal_model()

try:
    ensure_model_files()
    print('Model assets checked at startup.')
except Exception as exc:
    print(f'Warning: model assets were not available at startup: {exc}')

# Current year for footer
current_year = datetime.now().year


def model_metric():
    # Define the data dictionary
    model_metrics = {
        'Healthy': {
            'Precision': 0.94,
            'Recall': 0.94,
            'F1 Score': 0.94,
            'Support': 626
        },
        'Glaucomatous': {
            'Precision': 0.89,
            'Recall': 0.90,
            'F1 Score': 0.89,
            'Support': 345
        }
    }
    
    model_accuracy=0.9237899073120495*100
    lag_accuracy=97.20
    
    return model_metrics, model_accuracy, lag_accuracy

# Function to resize an image
def resize_image(image, size):
    resized_image = image.resize((size, size))
    return resized_image

# Function to check if an image is retinal
def check_image(img, model):
    claasss = predict_image(img, model)
    return claasss == 'Retinal'

# Define breadcrumbs data
def get_breadcrumbs():
    return [{'text': 'Dashboard', 'url': '/'}]

@app.route('/')
def dashboard():
    breadcrumbs = get_breadcrumbs()
    model_metrics, accuracy, lag = model_metric()
    return render_template('dashboard.html', breadcrumbs=breadcrumbs, year=current_year, model_metrics=model_metrics, accuracy=accuracy, lag=lag)

@app.route('/image1')
def get_image1():
    return send_from_directory('images', 'download.png')

@app.route('/image2')
def get_image2():
    return send_from_directory('images', 'download (1).png')

@app.route('/image3')
def get_image3():
    return send_from_directory('images', 'download (2).png')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    breadcrumbs = get_breadcrumbs()
    breadcrumbs.append({'text': 'Upload Image', 'url': '/upload_image'})
    
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('Please select an image to upload.')
            return redirect(url_for('upload_image'))

        if not allowed_file(file.filename):
            flash('Invalid file type. Upload a PNG or JPEG image.')
            return redirect(url_for('upload_image'))

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)

        try:
            image = Image.open(file_path)
            image.verify()
            image = Image.open(file_path)
        except Exception:
            flash('Uploaded file is not a valid image.')
            return redirect(url_for('upload_image'))

        try:
            model1 = get_retinal_model()
            if not check_image(image, model1):
                flash('Please upload a valid retinal image to continue!')
                return redirect(url_for('upload_image'))
        except Exception as e:
            flash(f'Error loading model: {str(e)}')
            return redirect(url_for('upload_image'))

        return redirect(url_for('display_prediction', filename=filename))
    
    # Render the upload page
    return render_template('upload_image.html', breadcrumbs=breadcrumbs, year=current_year)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/prediction/<filename>')
def display_prediction(filename):
    breadcrumbs = get_breadcrumbs()
    breadcrumbs.append({'text': 'Upload Image', 'url': '/upload_image'})
    breadcrumbs.append({'text': 'Prediction', 'url': '/prediction/{}'.format(filename)})
    
    image_url = url_for('uploaded_file', filename=filename)

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(image_path):
        flash('Uploaded image not found. Please upload again.')
        return redirect(url_for('upload_image'))

    try:
        image = Image.open(image_path)
    except Exception:
        flash('Unable to load the uploaded image.')
        return redirect(url_for('upload_image'))

    image = resize_image(image, 224)
    
    try:
        model = get_glaucoma_model()
        predictions, claas = predict(model, image)
    except Exception as e:
        flash(f'Error making prediction: {str(e)}')
        return redirect(url_for('upload_image'))

    return render_template('prediction_display.html', predictions=predictions, classs=claas, filename=filename, image_url=image_url, breadcrumbs=breadcrumbs, year=current_year)
    
if __name__ == '__main__':
    app.run(debug=True)