import numpy as np
from django.shortcuts import render
import requests
from django.conf import settings

def format_features(model_name, raw_features):
    """Convert all features to appropriate type based on model"""
    try:
        if model_name == 'breast_cancer':
            return [float(x) for x in raw_features]  # 30 float features
        elif model_name == 'diabetes':
            return [int(x) for x in raw_features]    # 16 int features
        elif model_name in ['heart', 'heart1']:
            return [float(x) for x in raw_features]  # 13 float features
        elif model_name in ['disease_dtc', 'disease_gnb']:
            return [int(x) for x in raw_features]    # int features
        return raw_features
    except ValueError as e:
        raise ValueError(f"Error converting features: {str(e)}")

def make_prediction(model_name, features):
    """Make API call to ML service"""
    try:
        response = requests.post(
            f"{settings.ML_SERVICE_URL}/predict",
            json={
                'model_name': model_name,
                'features': features
            },
            timeout=10  # Add timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"ML service error: {response.text}")  # For debugging
            raise Exception(f"ML service error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")  # For debugging
        raise Exception("Failed to connect to ML service")

def home(request):
    return render(request, 'home.html')

def diabetes(request):
    return render(request, 'diabetes.html')

def diabetes_result(request):
    try:
        features = [
            request.GET['age'],
            request.GET['gender'],
            request.GET['polyuria'],
            request.GET['polydipsia'],
            request.GET['suddenWeightLoss'],
            request.GET['Weakness'],
            request.GET['polyphagia'],
            request.GET['genitalThrush'],
            request.GET['visualBlurring'],
            request.GET['itching'],
            request.GET['irritability'],
            request.GET['delayedHealing'],
            request.GET['partialParesis'],
            request.GET['muscleStiffness'],
            request.GET['alopecia'],
            request.GET['obesity']
        ]
        
        processed_features = format_features('diabetes', features)
        result = make_prediction('diabetes', processed_features)
        
        dia_pred = result['prediction'][0]  # Get first element of prediction array
        
        if dia_pred == 0:
            return render(request, 'negative.html', {
                'pred': "NEGATIVE",
                'word1': "We are happy to tell you that you have tested negative for diabetes"
            })
        else:
            return render(request, 'positive.html', {
                'pred': "POSITIVE",
                'word1': "We are sorry to tell you that you have tested positive for diabetes",
                'word2': "Please consult an Endocrinologist"
            })
            
    except Exception as e:
        print(f"Error in diabetes prediction: {str(e)}")  # For debugging
        return render(request, 'error.html', {'error': str(e)})

def heart_disease_result(request):
    try:
        features = [
            request.GET['age'],
            request.GET['sex'],
            request.GET['cp'],
            request.GET['trestbps'],
            request.GET['chol'],
            request.GET['fbs'],
            request.GET['restecg'],
            request.GET['thalach'],
            request.GET['exang'],
            request.GET['oldpeak'],
            request.GET['slope'],
            request.GET['ca'],
            request.GET['thal']
        ]
        
        processed_features = format_features('heart', features)
        result = make_prediction('heart', processed_features)
        
        heart_pred = result['prediction'][0]  # Get first element of prediction array
        
        if heart_pred == 0:
            message = 'we are happy to tell you that you are less prone to heart disease'
            return render(request, 'negative.html', {'pred': message})
        else:
            message = 'we are sorry to tell you that you are more prone to heart disease'
            return render(request, 'positive.html', {'pred': message})
            
    except Exception as e:
        print(f"Error in heart disease prediction: {str(e)}")  # For debugging
        return render(request, 'error.html', {'error': str(e)})

def breast_cancer_result(request):
    try:
        features = [
            request.GET['radiusMean'], request.GET['textureMean'], request.GET['perimeterMean'],
            request.GET['areaMean'], request.GET['smoothnessMean'], request.GET['compactnessMean'],
            request.GET['concavityMean'], request.GET['concavePointsMean'], request.GET['symmetryMean'],
            request.GET['fractalDimensionMean'], request.GET['radiusSe'], request.GET['textureSe'],
            request.GET['perimeterSe'], request.GET['areaSe'], request.GET['smoothnessSe'],
            request.GET['compactnessSe'], request.GET['concavitySe'], request.GET['concavePointsSe'],
            request.GET['symmetrySe'], request.GET['fractalDimensionSe'], request.GET['radiusWorst'],
            request.GET['textureWorst'], request.GET['perimeterWorst'], request.GET['areaWorst'],
            request.GET['smoothnessWorst'], request.GET['compactnessWorst'], request.GET['concavityWorst'],
            request.GET['concavePointsWorst'], request.GET['symmetryWorst'], request.GET['fractalDimensionWorst']
        ]
        
        processed_features = format_features('breast_cancer', features)
        result = make_prediction('breast_cancer', processed_features)
        
        cancer_pred = result['prediction'][0]  # Get first element of prediction array
        
        if cancer_pred == 0:
            message = 'we have predicted that the cancer is benign'
            return render(request, 'negative.html', {'pred': message})
        else:
            message = 'we have predicted that the cancer is malignant'
            return render(request, 'positive.html', {'pred': message})
            
    except Exception as e:
        print(f"Error in breast cancer prediction: {str(e)}")  # For debugging
        return render(request, 'error.html', {'error': str(e)})
