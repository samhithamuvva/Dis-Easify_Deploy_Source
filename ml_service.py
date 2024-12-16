from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional
import numpy as np
from joblib import load
import tensorflow as tf
import os
import traceback
import base64
from PIL import Image
import io

app = FastAPI()

# Print environment information
print("Environment Information:")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

# Load all models at startup
models = {}

# Load joblib models configuration
joblib_models = {
    'breast_cancer': 'breast_cancer_rfc_model.joblib',
    'diabetes': 'diabetes_dtc_model.joblib',
    'disease_dtc': 'disease_dtc_model.joblib',
    'disease_gnb': 'disease_gnb_model.joblib',
    'heart': 'heart_rfc_model.joblib',
    'heart1': 'heart_rfc_model1.joblib'
}

# Load TensorFlow models configuration
tf_models = {
    'pneumonia': 'pneumonia.h5'
}

# Check savedModels directory
savedModels_path = '/savedModels'
print(f"\nChecking savedModels directory:")
print(f"savedModels path exists: {os.path.exists(savedModels_path)}")
if os.path.exists(savedModels_path):
    print(f"savedModels contents: {os.listdir(savedModels_path)}")

# Load joblib models
print("\nLoading joblib models:")
for model_name, model_file in joblib_models.items():
    try:
        model_path = os.path.join(savedModels_path, model_file)
        print(f"Loading {model_name} from {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        
        models[model_name] = load(model_path)
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

# Load TensorFlow models
print("\nLoading TensorFlow models:")
for model_name, model_file in tf_models.items():
    try:
        model_path = os.path.join(savedModels_path, model_file)
        print(f"Loading {model_name} from {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            print(f"Error: {model_path} does not exist!")
            continue
            
        # Try loading the model
        print("Attempting to load model...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        models[model_name] = model
        print(f"Successfully stored {model_name} in models dictionary")
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

class PredictionRequest(BaseModel):
    model_name: str
    features: List[Union[float, int]]
    image: Optional[str] = None

@app.post("/predict")
async def predict(request: PredictionRequest):
    print(f"\nReceived prediction request for model: {request.model_name}")
    
    if request.model_name not in models:
        print(f"Model {request.model_name} not found. Available models: {list(models.keys())}")
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    try:
        model = models[request.model_name]
        
        # Special handling for pneumonia model
        if request.model_name == 'pneumonia' and request.image:
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(request.image)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Preprocess image
                image = image.convert('L')  # Convert to grayscale
                image = image.resize((150, 150))  # Resize to expected dimensions
                img_array = np.array(image)
                img_array = img_array / 255.0  # Normalize
                img_array = img_array.reshape((1, 150, 150, 1))  # Reshape for model
                
                # Make prediction
                prediction = model.predict(img_array)
                return {
                    "prediction": int(prediction[0][0] > 0.5),
                    "confidence": float(prediction[0][0])
                }
                
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
        
        # Handle other models
        features = np.array(request.features)
        print(f"Feature shape: {features.shape}")
        
        # Make prediction
        prediction = model.predict(features.reshape(1, -1))
        print(f"Prediction result: {prediction}")
        
        return {"prediction": prediction.tolist()}
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
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

# Print loaded models summary
print("\nLoaded Models Summary:")
print(f"Total models loaded: {len(models)}")
print(f"Available models: {list(models.keys())}")
