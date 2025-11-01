# utils.py

import tensorflow as tf
from tensorflow import keras
from PIL import Image
from io import BytesIO
import numpy as np

# --- Configuration: Pointing to the Subdirectory ---
# NOTE: The path is 'model/' because the Dockerfile copied the 'model' folder here.
MODEL_PATH = 'model/waste_model.h5'
LABEL_FILE = 'model/labels.txt'
TARGET_SIZE = (224, 224) 
# The normalization is based on your training code: rescale=1./255

# Define which classes count as a successful recycling action for Aptos
# Based on your labels:
RECYCLABLE_CLASSES = [
    "battery", "brown-glass", "cardboard", "green-glass", 
    "metal", "paper", "plastic", "white-glass", "clothes", "shoes" 
] 
# Note: 'biological' and 'trash' are excluded. All labels are converted to lowercase.

# --- Global Model Loading (Occurs when the API container starts) ---
try:
    # Use compile=False if you are having issues loading MobileNetV2 weights
    WASTE_MODEL = keras.models.load_model(MODEL_PATH)
    
    # Load labels: one label per line, convert to lowercase for robust checking
    with open(LABEL_FILE, 'r') as f:
        LABELS = [line.strip().lower() for line in f]
    
    MODEL_STATUS = "Ready"
    print("AI Model and labels loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model or labels at startup: {e}")
    WASTE_MODEL = None
    LABELS = []
    MODEL_STATUS = "Error"


def preprocess_and_predict(file_data: bytes):
    """
    Handles image preprocessing and runs model prediction.
    """
    if WASTE_MODEL is None:
        raise ValueError("Model is not loaded. Cannot predict.")

    # 1. Read, Format, and Resize Image
    img = Image.open(BytesIO(file_data)).convert('RGB')
    img = img.resize(TARGET_SIZE) 
    
    # 2. Convert to NumPy Array and NORMALIZE (Matches training's rescale=1./255)
    img_array = keras.preprocessing.image.img_to_array(img)
    normalized_array = img_array / 255.0 
    
    # 3. Add batch dimension (TensorFlow expects a batch: (1, H, W, C))
    final_input = np.expand_dims(normalized_array, axis=0) 

    # 4. Predict and interpret results
    predictions = WASTE_MODEL.predict(final_input)
    
    # Get highest probability score and index
    score = np.max(predictions)
    class_index = np.argmax(predictions)
    
    # Look up the human-readable class name (ensuring it's lowercase)
    predicted_class = LABELS[class_index]
    
    # --- Verification Logic ---
    is_verified_recyclable = predicted_class in RECYCLABLE_CLASSES
    
    # Return the three necessary outputs
    return predicted_class, float(score), is_verified_recyclable
