from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from joblib import load
import tensorflow as tf
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Disease Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Print current directory and contents for debugging
print("Current directory:", os.getcwd())
print("Directory contents:", os.listdir())
print("Models directory contents:", os.listdir("/savedModels"))

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
        logger.info(f"Attempting to load {model_name} from /savedModels/{model_file}")
        models[model_name] = load(f'/savedModels/{model_file}')
        logger.info(f"Loaded {model_name} successfully")
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")

# Load TensorFlow models
for model_name, model_file in tf_models.items():
    try:
        logger.info(f"Attempting to load {model_name} from /savedModels/{model_file}")
        models[model_name] = tf.keras.models.load_model(f'/savedModels/{model_file}')
        logger.info(f"Loaded {model_name} successfully")
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")

class PredictionRequest(BaseModel):
    model_name: str
    features: List[Union[float, int]]

@app.get("/")
async def root():
    """Root endpoint to indicate API is live."""
    return {"status": "OK", "message": "Welcome to the Disease Prediction API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "available_models": list(models.keys()),
        "model_count": len(models)
    }

@app.get("/models")
async def list_models():
    """List all available models and their details."""
    return {
        "total_models": len(models),
        "available_models": list(models.keys()),
        "models_info": {
            "joblib_models": list(joblib_models.keys()),
            "tensorflow_models": list(tf_models.keys())
        }
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions using the specified model."""
    if request.model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model {request.model_name} not found. Available models: {list(models.keys())}"
        )
    
    try:
        model = models[request.model_name]
        features = np.array(request.features)
        
        # Handle TensorFlow models differently
        if request.model_name in tf_models:
            # Add specific shape handling for pneumonia model
            if request.model_name == 'pneumonia':
                features = features.reshape((1, 36, 36, 1))  # Adjust shape as needed
            prediction = model.predict(features)
        else:
            prediction = model.predict(features.reshape(1, -1))
        
        return {
            "model_name": request.model_name,
            "prediction": prediction.tolist(),
            "status": "success"
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input shape: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
