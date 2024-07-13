import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Load pre-trained ResNet50 model trained on ImageNet (for feature extraction)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Preprocess function for a single image
def preprocess_image(image_input, target_size=(224, 224)):
    image_input = np.array(image_input)
    
     # Ensure the image has 3 channels
    if image_input.shape[-1] == 1:
        # Convert grayscale to RGB
        image_input = np.repeat(image_input, 3, axis=-1)
    elif image_input.shape[-1] == 2:
        # Convert 2-channel to 3-channel by repeating one channel
        image_input = np.concatenate((image_input, image_input[:, :, :1]), axis=-1)
    elif image_input.shape[-1] == 4:
        # Convert 4-channel to 3-channel by dropping the alpha channel
        image_input = image_input[:, :, :3]
    
    # Resize the image to the target size
    image_resized = tf.image.resize(image_input, target_size).numpy()
    
    # Add a batch dimension
    image_batch = np.expand_dims(image_resized, axis=0)
    
    # Normalize pixel values
    image_batch /= 255.0
    
    return image_batch

# Function to predict the image class
def predict_image(image_input, model):
    # Preprocess the input image
    img_array = preprocess_image(image_input)

    # Extract features using the ResNet50 base model
    features = base_model.predict(img_array, verbose=0)
    features = features.reshape((features.shape[0], -1))

    # Make predictions using the trained model
    prediction = model.predict(features, verbose=0)

    # Determine the class based on the prediction threshold
    classs = 'Retinal' if prediction[0][0] > 0.5 else 'Non-Retinal'
    
    return classs