import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from PIL import Image
from datetime import datetime
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from model import load_modell, predict
from retinal import predict_image
from db import init_db, is_database_empty, authenticate, add_user
from config import MODEL_PATH

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load your models
model = load_modell(MODEL_PATH)
model1 = load_model('models/retinal.h5')

# Current year for footer
current_year = datetime.now().year

# Define a directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
MASK_FOLDER = 'masks'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER

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
    breadcrumbs = [{'text': 'Logout', 'url': '/logout'}]
    if 'username' in session:
        breadcrumbs.append({'text': 'Dashboard', 'url': '/'})
    return breadcrumbs

@app.route('/register', methods=['GET', 'POST'])
def register():
    if is_database_empty():
        init_db()  # Initialize database if it's empty

    if request.method == 'POST':
        full_name = request.form['full_name']
        username = request.form['username']
        password = request.form['password']
        # Add user to the database
        add_user(full_name, username, password)
        return redirect(url_for('login'))

    return render_template('register.html', year=current_year)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_database_empty():
        return redirect(url_for('register'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = authenticate(username, password)
        if user:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html', year=current_year)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
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
    if 'username' not in session:
        return redirect(url_for('login'))
    
    breadcrumbs = get_breadcrumbs()
    breadcrumbs.append({'text': 'Upload Image', 'url': '/upload_image'})
    
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file to the uploads directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Check if the uploaded image is retinal
            image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_array = np.array(image)

            if not check_image(image, model1):
                flash('Please upload a valid retinal image to continue!')
                return redirect(url_for('upload_image'))

            # Redirect to the prediction display page
            return redirect(url_for('display_prediction', filename=filename))
    
    # Render the upload page
    return render_template('upload_image.html', breadcrumbs=breadcrumbs, year=current_year)

@app.route('/prediction/<filename>')
def display_prediction(filename):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    breadcrumbs = get_breadcrumbs()
    breadcrumbs.append({'text': 'Upload Image', 'url': '/upload_image'})
    breadcrumbs.append({'text': 'Prediction', 'url': '/prediction/{}'.format(filename)})
    
    image_url = url_for('static', filename=os.path.join('uploads', filename))

    # Construct the URL for the uploaded image
    image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    image = resize_image(image, 224)
    img_array = np.array(image)

    # Normalize pixel values if necessary
    img_array = img_array / 255.0  # Normalize to [0, 1] range

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Perform prediction using the loaded model
    predictions, claas = predict(model, image)

    return render_template('prediction_display.html', predictions=predictions, classs=claas, filename=filename, image_url=image_url, breadcrumbs=breadcrumbs, year=current_year)
    
if __name__ == '__main__':
    app.run(debug=True)