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

# TensorFlow models configuration with expected shapes
tf_models = {
    'pneumonia': {
        'file': 'pneumonia.h5',
        'input_shape': (224, 224, 1)  # Standard shape for medical images
    }
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
for model_name, model_info in tf_models.items():
    try:
        logger.info(f"Attempting to load {model_name} from /savedModels/{model_info['file']}")
        # Custom load for TensorFlow models
        custom_objects = None
        models[model_name] = tf.keras.models.load_model(
            f'/savedModels/{model_info["file"]}',
            custom_objects=custom_objects,
            compile=False
        )
        # Recompile the model
        models[model_name].compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.info(f"Loaded {model_name} successfully")
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")

class PredictionRequest(BaseModel):
    model_name: str
    features: List[Union[float, int]]

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
        
        # Handle different model types
        if request.model_name in tf_models:
            # For pneumonia model
            expected_shape = tf_models[request.model_name]['input_shape']
            features = features.reshape((1,) + expected_shape)
            # Ensure values are between 0 and 1
            features = features / 255.0 if features.max() > 1.0 else features
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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "available_models": list(models.keys()),
        "model_count": len(models),
        "loaded_models": {
            name: "loaded" if name in models else "failed"
            for name in list(joblib_models.keys()) + list(tf_models.keys())
        }
    }

@app.get("/models")
async def list_models():
    """List all available models and their details."""
    return {
        "total_models": len(models),
        "available_models": list(models.keys()),
        "models_info": {
            "joblib_models": {
                name: "loaded" for name in joblib_models.keys() if name in models
            },
            "tensorflow_models": {
                name: {
                    "status": "loaded" if name in models else "failed",
                    "expected_input_shape": model_info["input_shape"]
                }
                for name, model_info in tf_models.items()
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
