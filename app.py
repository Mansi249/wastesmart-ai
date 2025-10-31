from flask import Flask, request, render_template, jsonify
import os
import tensorflow as tf
from utils.preprocess import preprocess_image
from utils.predict import predict_image

app = Flask(__name__)

# Load model and labels
MODEL_PATH = "model/waste_model.h5"
LABELS_PATH = "model/labels.txt"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return "WasteSmart-AI Flask App is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    img_array = preprocess_image(file_path)
    predicted_label, confidence = predict_image(model, img_array, labels)

    os.remove(file_path)
    return jsonify({
        "prediction": predicted_label,
        "confidence": f"{confidence}%"
    })

if __name__ == "__main__":
    app.run(debug=True)
