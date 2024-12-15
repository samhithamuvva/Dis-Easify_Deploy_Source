from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from joblib import load
import tensorflow as tf
import os
import sys
import traceback

app = FastAPI()

# Enhanced Logging Function
def log_error(context, exception):
    print(f"ERROR in {context}:")
    print(f"Error Type: {type(exception).__name__}")
    print(f"Error Details: {str(exception)}")
    print("Full Traceback:")
    traceback.print_exc()

# Comprehensive System and Environment Check
print("Comprehensive System Information:")
print("Current Working Directory:", os.getcwd())
print("Python Version:", sys.version)
print("TensorFlow Version:", tf.__version__)

# Verify savedModels Directory
savedModels_path = 'savedModels'
try:
    print("\nSavedModels Directory Check:")
    print(f"Directory Exists: {os.path.exists(savedModels_path)}")
    print(f"Is Directory: {os.path.isdir(savedModels_path)}")
    print("Contents:", os.listdir(savedModels_path))
except Exception as e:
    log_error("SavedModels Directory Check", e)

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
        full_path = os.path.join(savedModels_path, model_file)
        print(f"\nLoading Joblib Model {model_name}:")
        print(f"Full Path: {full_path}")
        print(f"File Exists: {os.path.exists(full_path)}")
        
        if not os.path.exists(full_path):
            print(f"WARNING: {full_path} does not exist!")
            continue
        
        models[model_name] = load(full_path)
        print(f"Loaded Joblib Model {model_name} successfully")
    except Exception as e:
        log_error(f"Joblib Model {model_name} Loading", e)

# Enhanced TensorFlow Model Loading
for model_name, model_file in tf_models.items():
    try:
        full_path = os.path.join(savedModels_path, model_file)
        
        print(f"\nAttempting to Load TensorFlow Model {model_name}:")
        print(f"Full Path: {full_path}")
        print(f"Absolute Path: {os.path.abspath(full_path)}")
        
        # Comprehensive File Checks
        if not os.path.exists(full_path):
            print(f"ERROR: File {full_path} does not exist!")
            continue
        
        if not os.path.isfile(full_path):
            print(f"ERROR: {full_path} is not a valid file!")
            continue
        
        # File Size Check
        file_size = os.path.getsize(full_path)
        print(f"File Size: {file_size} bytes")
        
        if file_size == 0:
            print(f"ERROR: {full_path} is an empty file!")
            continue
        
        # Alternative Loading Methods
        loading_methods = [
            lambda: tf.keras.models.load_model(full_path, compile=False),
            lambda: tf.keras.saving.load_model(full_path, compile=False)
        ]
        
        model_loaded = False
        for method in loading_methods:
            try:
                model = method()
                print(f"Model {model_name} loaded successfully")
                print(f"Input Shape: {model.input_shape}")
                print(f"Output Shape: {model.output_shape}")
                
                models[model_name] = model
                model_loaded = True
                break
            except Exception as load_error:
                print(f"Loading method failed: {str(load_error)}")
        
        if not model_loaded:
            print(f"FAILED to load TensorFlow Model {model_name}")
    
    except Exception as e:
        log_error(f"TensorFlow Model {model_name} Loading", e)

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
