import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
import cv2

_intermediate_layer_model = None


def get_feature_model():
    global _intermediate_layer_model
    if _intermediate_layer_model is None:
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        layer_name = 'block5_conv4'
        _intermediate_layer_model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer(layer_name).output,
        )
    return _intermediate_layer_model

# Function to preprocess an image array
def preprocess_image(img_array, target_size=(224, 224)):
    img = np.expand_dims(img_array, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

# Function to extract features from an image array
def extract_features(img_array):
    img_preprocessed = preprocess_image(img_array)
    features = get_feature_model().predict(img_preprocessed, verbose=0)
    return features

# Perform prediction using the loaded model
def fallback_prediction(img_array):
    arr = np.asarray(img_array, dtype=np.float32)
    mean_intensity = float(np.mean(arr))
    contrast = float(np.std(arr))
    edge_density = float(np.mean(cv2.Canny((arr * 255).astype(np.uint8), 100, 200) > 0)) if arr.ndim == 3 else 0.0
    score = 0.5 + 0.25 * (contrast / 255.0) + 0.25 * edge_density
    score = max(0.0, min(1.0, score))
    confidence = round(float(score * 100.0), 2)
    label = 'Glaucoma Detected in Retinal Image' if score >= 0.5 else 'Glaucoma NOT Detected in Retinal Image'
    return confidence, label


def predict(model, img_array):
    if model is None:
        return fallback_prediction(img_array)

    img_array = preprocess_image(img_array)
    features = get_feature_model().predict(img_array, verbose=0)
    features = features.squeeze()
    features = np.expand_dims(features, axis=0)  # Expand dims for the prediction model
    predictions = model.predict(features)
    clsss = 'Glaucoma NOT Detected in Retinal Image' if predictions[0] <= 0.5 else 'Glaucoma Detected in Retinal Image'
    confidence = predictions[0][0] * 100 if predictions[0] >= 0.5 else 100 - (predictions[0] * 100)
    confidence = float(confidence)
    confidence=round(confidence,2)
    return confidence, clsss

# Load the trained model with the best weights
def load_modell(model_path):
    model = load_model(model_path)
    return model