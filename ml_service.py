from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from joblib import load
import tensorflow as tf
import os

app = FastAPI()

# Load all models at startup
models = {}

# Load joblib models
joblib_models = {
    'breast_cancer': 'breast_cancer_rfc_model.joblib',
    'diabetes': 'diabetes_dtc_model.joblib',
    'disease_dtc': 'disease_dtc_model.joblib',
    'disease_gnb': 'disease_gnb_model.joblib',
    'heart': 'heart_rfc_model.joblib',
    'heart1': 'heart_rfc_model1.joblib'
}

# Load TensorFlow models
tf_models = {
    'pneumonia': 'pneumonia.h5'
}

# Load joblib models
for model_name, model_file in joblib_models.items():
    try:
        models[model_name] = load(f'/app/models/{model_file}')
        print(f"Loaded {model_name} successfully")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

# Load TensorFlow models
for model_name, model_file in tf_models.items():
    try:
        models[model_name] = tf.keras.models.load_model(f'/app/models/{model_file}')
        print(f"Loaded {model_name} successfully")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

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
        
        # Reshape features for TensorFlow models if needed
        if request.model_name in tf_models:
            features = features.reshape(model.input_shape)
        
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
