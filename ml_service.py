from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional
import numpy as np
from joblib import load
import tensorflow as tf
import os
import sys
import traceback
import base64
import logging
from io import BytesIO
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ml_service.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enhanced Logging Function
def log_error(context, exception):
    logger.error(f"ERROR in {context}:")
    logger.error(f"Error Type: {type(exception).__name__}")
    logger.error(f"Error Details: {str(exception)}")
    logger.error("Full Traceback:")
    logger.error(traceback.format_exc())

# Comprehensive System and Environment Check
logger.info("Comprehensive System Information:")
logger.info(f"Current Working Directory: {os.getcwd()}")
logger.info(f"Python Version: {sys.version}")
logger.info(f"TensorFlow Version: {tf.__version__}")

# Verify savedModels Directory
savedModels_path = os.path.join(os.getcwd(), 'savedModels')
try:
    logger.info("\nSavedModels Directory Check:")
    logger.info(f"Directory Exists: {os.path.exists(savedModels_path)}")
    logger.info(f"Is Directory: {os.path.isdir(savedModels_path)}")
    logger.info(f"Contents: {os.listdir(savedModels_path)}")
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
        logger.info(f"\nLoading Joblib Model {model_name}:")
        logger.info(f"Full Path: {full_path}")
        logger.info(f"File Exists: {os.path.exists(full_path)}")
        
        if not os.path.exists(full_path):
            logger.warning(f"WARNING: {full_path} does not exist!")
            continue
        
        models[model_name] = load(full_path)
        logger.info(f"Loaded Joblib Model {model_name} successfully")
    except Exception as e:
        log_error(f"Joblib Model {model_name} Loading", e)

# Enhanced TensorFlow Model Loading
for model_name, model_file in tf_models.items():
    try:
        full_path = os.path.join(savedModels_path, model_file)
        
        logger.info(f"\nAttempting to Load TensorFlow Model {model_name}:")
        logger.info(f"Full Path: {full_path}")
        logger.info(f"Absolute Path: {os.path.abspath(full_path)}")
        
        # Comprehensive File Checks
        if not os.path.exists(full_path):
            logger.error(f"ERROR: File {full_path} does not exist!")
            continue
        
        if not os.path.isfile(full_path):
            logger.error(f"ERROR: {full_path} is not a valid file!")
            continue
        
        # File Size Check
        file_size = os.path.getsize(full_path)
        logger.info(f"File Size: {file_size} bytes")
        
        if file_size == 0:
            logger.error(f"ERROR: {full_path} is an empty file!")
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
                logger.info(f"Model {model_name} loaded successfully")
                logger.info(f"Input Shape: {model.input_shape}")
                logger.info(f"Output Shape: {model.output_shape}")
                
                models[model_name] = model
                model_loaded = True
                break
            except Exception as load_error:
                logger.error(f"Loading method failed: {str(load_error)}")
        
        if not model_loaded:
            logger.error(f"FAILED to load TensorFlow Model {model_name}")
    
    except Exception as e:
        log_error(f"TensorFlow Model {model_name} Loading", e)

class PredictionRequest(BaseModel):
    model_name: str
    features: List[Union[float, int]]
    image: Optional[str] = None

@app.post("/predict")
async def predict(request: PredictionRequest):
    logger.info(f"Received prediction request for: {request.model_name}")
    
    if request.model_name not in models:
        logger.error(f"Model not found: {request.model_name}")
        logger.error(f"Available models: {list(models.keys())}")
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    try:
        model = models[request.model_name]
        
        # Pneumonia Model Special Handling
        if request.model_name == 'pneumonia' and request.image:
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(request.image)
                image = Image.open(BytesIO(image_bytes)).convert('L')  # Convert to grayscale
                image = image.resize((150, 150))  # Assuming model expects 150x150
                
                # Normalize and reshape
                img_array = np.array(image) / 255.0
                img_array = img_array.reshape((1, 150, 150, 1))
                
                prediction = model.predict(img_array)
                return {"prediction": int(prediction[0][0] > 0.5), "confidence": float(prediction[0][0])}
            
            except Exception as image_error:
                logger.error(f"Image processing error: {str(image_error)}")
                raise HTTPException(status_code=400, detail="Invalid image")
        
        # Other models
        features = np.array(request.features)
        
        if request.model_name in tf_models:
            # Reshape to match model's expected input
            features = features.reshape(model.input_shape[1:])
        
        prediction = model.predict(features.reshape(1, -1))
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        logger.error(f"Prediction Error for {request.model_name}:")
        logger.error(str(e))
        logger.error(traceback.format_exc())
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

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "An unexpected error occurred"}
