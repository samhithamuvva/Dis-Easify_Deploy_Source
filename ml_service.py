from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from joblib import load
import tensorflow as tf
import os
import logging
import base64
from PIL import Image
import io

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
    'disease_gnb': 'disease_gnb_model.joblib',
    'heart': 'heart_dtc_model1.joblib',
}

# TensorFlow models configuration with expected shapes
tf_models = {
    'pneumonia': {
        'file': 'pneumonia2.h5',
        'input_shape': (36, 36, 1)  # Updated to match the error message
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
        
        # Define the input shape explicitly
        input_shape = model_info['input_shape']
        inputs = tf.keras.Input(shape=input_shape)
        
        # Try loading the model with custom input shape
        try:
            base_model = tf.keras.models.load_model(
                f'/savedModels/{model_info["file"]}',
                compile=False
            )
            # Rebuild the model with correct input shape
            x = inputs
            for layer in base_model.layers[1:]:  # Skip the input layer
                x = layer(x)
            model = tf.keras.Model(inputs=inputs, outputs=x)
            
        except Exception as model_error:
            logger.error(f"Error rebuilding model: {model_error}")
            # Try alternative model if available
            try:
                logger.info("Attempting to load pneumonia2.h5")
                model = tf.keras.models.load_model(f'/savedModels/pneumonia2.h5', compile=False)
            except Exception as alt_error:
                logger.error(f"Error loading alternative model: {alt_error}")
                raise
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        models[model_name] = model
        logger.info(f"Loaded {model_name} successfully")
        
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")
        # Try the next alternative
        try:
            logger.info("Attempting to load pneumonia3.h5 as final alternative")
            model = tf.keras.models.load_model(f'/savedModels/pneumonia3.h5', compile=False)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            models[model_name] = model
            logger.info(f"Loaded {model_name} (using pneumonia3.h5) successfully")
        except Exception as final_error:
            logger.error(f"Failed to load any pneumonia model: {final_error}")

class PredictionRequest(BaseModel):
    model_name: str
    features: List[Union[float, int]]

class ImagePredictionRequest(BaseModel):
    model_name: str
    image: str  # base64 encoded image

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
            input_shape = tf_models[request.model_name]['input_shape']
            features = features.reshape((-1,) + input_shape)
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

@app.post("/predict_image")
async def predict_image(request: ImagePredictionRequest):
    """Endpoint for image-based predictions"""
    if request.model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model {request.model_name} not found"
        )
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale and resize
        image = image.convert('L')
        image = image.resize((36, 36))
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = image_array.reshape(1, 36, 36, 1)
        
        # Make prediction
        prediction = models[request.model_name].predict(image_array)
        confidence = float(prediction[0][0]) * 100
        
        return {
            "prediction": int(prediction[0][0] > 0.5),
            "confidence": round(confidence, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

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
                    "expected_input_shape": model_info["input_shape"] if name in models else None
                }
                for name, model_info in tf_models.items()
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
