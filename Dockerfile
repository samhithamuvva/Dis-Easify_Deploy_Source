FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY savedModels/ /app/savedModels/
COPY ml_service.py .

# Verify model files
RUN python3 -c "\
import os; \
import tensorflow as tf; \
print('Current Directory:', os.getcwd()); \
print('SavedModels Path:', os.path.abspath('savedModels')); \
print('SavedModels Contents:', os.listdir('savedModels')); \
model_path = os.path.join('savedModels', 'pneumonia.h5'); \
print('Model Path:', os.path.abspath(model_path)); \
model = tf.keras.models.load_model(model_path, compile=False); \
print('Pneumonia Model Loaded Successfully')"


EXPOSE 8080

CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "8080"]
