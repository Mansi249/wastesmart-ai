import tensorflow as tf
import numpy as np

def predict_image(model, img_array, labels):
    """
    Predicts class and confidence for an image.
    """
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = labels[predicted_index]
    confidence = round(float(np.max(predictions[0])) * 100, 2)
    return predicted_label, confidence
