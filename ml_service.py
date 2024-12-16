from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from joblib import load
import tensorflow as tf
import os

app = FastAPI()

# Print current directory and contents for debugging
print(f"Current directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

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

# Check if models directory exists
if not os.path.exists('/savedModels'):
    print("Error: /savedModels directory not found!")
    print(f"Available directories: {os.listdir('/')}")
else:
    print(f"Models directory contents: {os.listdir('/savedModels')}")

# Load joblib models
for model_name, model_file in joblib_models.items():
    try:
        file_path = f'/savedModels/{model_file}'
        print(f"Attempting to load {model_name} from {file_path}")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        models[model_name] = load(file_path)
        print(f"Loaded {model_name} successfully")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

# Load TensorFlow models
for model_name, model_file in tf_models.items():
    try:
        file_path = f'/savedModels/{model_file}'
        print(f"Attempting to load {model_name} from {file_path}")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        models[model_name] = tf.keras.models.load_model(file_path, compile=False)
        print(f"Loaded {model_name} successfully")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
