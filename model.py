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
def predict(model, img_array):
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