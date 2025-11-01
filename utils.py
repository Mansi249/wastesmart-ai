# utils.py
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from io import BytesIO
import numpy as np

# --- Configuration (relative paths inside Docker) ---
MODEL_PATH = 'waste_model.h5'
LABEL_FILE = 'labels.txt'

# Load model globally when the API starts (to prevent slow startup per request)
try:
    WASTE_MODEL = keras.models.load_model(MODEL_PATH)
    # Load labels: one label per line
    with open(LABEL_FILE, 'r') as f:
        LABELS = [line.strip() for line in f]
    MODEL_STATUS = "Ready"
    print("AI Model and labels loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model: {e}")
    WASTE_MODEL = None
    LABELS = []
    MODEL_STATUS = "Error"


def preprocess_and_predict(file_data: bytes, target_size=(224, 224)):
    """
    Handles image preprocessing and runs model prediction.

    Args:
        file_data: Raw bytes of the uploaded image file.
        target_size: The exact input size (Height, Width) the model expects.
    
    Returns: A tuple (predicted_class_name, confidence_score)
    """
    if WASTE_MODEL is None:
        raise ValueError("Model is not loaded. Cannot predict.")

    # 1. Read, Format, and Resize Image
    img = Image.open(BytesIO(file_data)).convert('RGB')
    img = img.resize(target_size) 
    
    # 2. Convert to NumPy Array and Normalize
    img_array = keras.preprocessing.image.img_to_array(img)
    # 🚨 CRITICAL: Ensure this normalization matches your Colab training!
    normalized_array = img_array / 255.0 
    
    # 3. Add batch dimension (1, H, W, C)
    final_input = np.expand_dims(normalized_array, axis=0) 

    # 4. Predict and interpret results
    predictions = WASTE_MODEL.predict(final_input)
    
    # Get highest probability score and index
    score = np.max(predictions)
    class_index = np.argmax(predictions)
    
    # Look up the human-readable class name
    predicted_class = LABELS[class_index]
    
    return predicted_class, float(score)
