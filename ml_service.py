from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from joblib import load
import tensorflow as tf
import os

app = FastAPI()

# Verbose TensorFlow and System Information
print("System Information:")
print("Current Working Directory:", os.getcwd())
print("TensorFlow Version:", tf.__version__)
print("Python Version:", import sys; sys.version)
print("Contents of savedModels:", os.listdir('savedModels'))

# Load models configuration
joblib_models = {
    'breast_cancer': 'breast_cancer_rfc_model.joblib',
    'diabetes': 'diabetes_dtc_model.joblib',
    'disease_dtc': 'disease_dtc_model.joblib',
    'disease_gnb': 'disease_gnb_model.joblib',
    'heart': 'heart_rfc_model.joblib',
    'heart1': 'heart_rfc_model1.joblib'
}

tf_models = {
    'pneumonia': 'pneumonia.h5'
}

# Initialize models dictionary
models = {}

# Load Joblib Models
for model_name, model_file in joblib_models.items():
    try:
        full_path = os.path.join('savedModels', model_file)
        models[model_name] = load(full_path)
        print(f"Loaded Joblib Model {model_name} successfully from {full_path}")
    except Exception as e:
        print(f"Error loading Joblib Model {model_name}: {e}")

# Load TensorFlow Models with Enhanced Debugging
for model_name, model_file in tf_models.items():
    try:
        full_path = os.path.join('savedModels', model_file)
        print(f"Attempting to load TensorFlow Model {model_name}")
        print(f"Full model path: {full_path}")
        print(f"Absolute path: {os.path.abspath(full_path)}")
        print(f"File exists: {os.path.exists(full_path)}")

        # Try loading with additional parameters
        model = tf.keras.models.load_model(full_path, compile=False)
        
        # Verify model loaded correctly
        print(f"Model {model_name} input shape: {model.input_shape}")
        print(f"Model {model_name} output shape: {model.output_shape}")
        
        models[model_name] = model
        print(f"Loaded TensorFlow Model {model_name} successfully")
    except Exception as e:
        print(f"Detailed Error Loading TensorFlow Model {model_name}:")
        print(str(e))
        import traceback
        traceback.print_exc()

class PredictionRequest(BaseModel):
    model_name: str
    features: List[Union[float, int]]

@app.post("/predict")
async def predict(request: PredictionRequest):
    if request.model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    try:
        model = models[request.model_name]
        features = np.array(request.features)
        
        # Special handling for TensorFlow models
        if request.model_name in tf_models:
            # Reshape to match model's expected input
            features = features.reshape(model.input_shape[1:])
        
        prediction = model.predict(features.reshape(1, -1))
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "available_models": list(models.keys())
    }

@app.get("/models")
async def list_models():
    return {
        "models": {
            "joblib_models": list(joblib_models.keys()),
            "tensorflow_models": list(tf_models.keys()),
            "loaded_models": list(models.keys())
        }
    }

# Optional: If you want to add more detailed error logging
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"Unhandled exception: {exc}")
    return {"error": "An unexpected error occurred"}
