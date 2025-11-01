# Dockerfile
#
# Use the official TensorFlow 2.15 GPU image (includes Linux, Python, CUDA, cuDNN)
FROM tensorflow/tensorflow:2.15.0-gpu

# Set the application directory inside the container
WORKDIR /app

# Copy the requirements file first for faster Docker caching
COPY requirements.txt .

# Install dependencies using pip (FastAPI, Uvicorn, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python logic files (app.py and utils.py)
COPY app.py .
COPY utils.py .

# Copy the entire model folder into the container's working directory (/app)
# This allows utils.py to access the model files via 'model/...'
COPY model /app/model/

# Expose the API port
EXPOSE 8000

# Start the Uvicorn web server
# The --host 0.0.0.0 makes the API accessible outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
