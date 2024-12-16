import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from joblib import load
import tensorflow as tf

app = FastAPI()

# Define base path for models
BASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'savedModels')

# Load all models at startup
models = {}

# Joblib models configuration
joblib_models = {
    'breast_cancer': 'breast_cancer_rfc_model.joblib',
    'diabetes': 'diabetes_dtc_model.joblib',
    'disease_dtc': 'disease_dtc_model.joblib',
    'disease_gnb': 'disease_gnb_model.joblib',
    'heart': 'heart_rfc_model.joblib',
    'heart1': 'heart_rfc_model1.joblib'
}

# TensorFlow models configuration
tf_models = {
    'pneumonia': 'pneumonia.h5'
}

# Load joblib models
for model_name, model_file in joblib_models.items():
    try:
        model_path = os.path.join(BASE_MODEL_PATH, model_file)
        models[model_name] = load(model_path)
        print(f"Loaded {model_name} successfully from {model_path}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

# Load TensorFlow models
for model_name, model_file in tf_models.items():
    try:
        model_path = os.path.join(BASE_MODEL_PATH, model_file)
        models[model_name] = tf.keras.models.load_model(model_path)
        print(f"Loaded {model_name} successfully from {model_path}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

@app.get("/")
async def root():
    """Root endpoint to indicate API is live."""
    return {"message": "Welcome to the ML Service API"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Endpoint to make predictions using the specified model."""
    if request.model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    try:
        model = models[request.model_name]
        features = np.array(request.features)

        # Handle TensorFlow models differently
        if request.model_name in tf_models:
            input_shape = model.input_shape[1:]  # Exclude batch size
            features = features.reshape(input_shape)
            prediction = model.predict(features[np.newaxis, ...])  # Add batch dimension
        else:
            prediction = model.predict(features.reshape(1, -1))
        
        return {"prediction": prediction.tolist()}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input shape: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running and models are loaded."""
    return {
        "status": "healthy",
        "available_models": list(models.keys())
    }

@app.get("/models")
async def list_models():
    """List all available models and their details."""
    return {
        "models": {
            "joblib_models": {name: "sklearn joblib" for name in joblib_models.keys()},
            "tensorflow_models": {
                name: models[name].input_shape if name in models else "Not Loaded"
                for name in tf_models.keys()
            },
            "loaded_models": list(models.keys())
        }
    }
