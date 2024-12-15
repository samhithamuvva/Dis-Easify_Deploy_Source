import numpy as np
from django.shortcuts import render
import requests
from django.conf import settings

# Removed local model loading since we'll use API calls now

# Home page
def home(request):
    return render(request, 'home.html')

# Diabetes Prediction
def diabetes(request):
    return render(request, 'diabetes.html')

def diabetes_result(request):
    try:
        # Collect all features from the request
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
        
        # Make API call to ML service
        response = requests.post(
            f"{settings.ML_SERVICE_URL}/predict", 
            json={
                'model_name': 'diabetes',
                'features': features
            }
        )
        
        if response.status_code == 200:
            dia_pred = response.json()['prediction']
            
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
        else:
            return render(request, 'error.html', {'error': 'ML service error'})
    
    except Exception as e:
        return render(request, 'error.html', {'error': str(e)})

# Heart Disease Prediction
def heart_disesase(request):
    return render(request, 'heart_css_pred.html')

def heart_disease_result(request):
    try:
        # Collect all features from the request
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
        
        # Make API call to ML service
        response = requests.post(
            f"{settings.ML_SERVICE_URL}/predict", 
            json={
                'model_name': 'heart_disease',
                'features': features
            }
        )
        
        if response.status_code == 200:
            heart_pred = response.json()['prediction']
            
            if heart_pred == 0:
                heart_pred = 'we are happy to tell you that you are less prone to heart disease'
                return render(request, 'negative.html', {'pred': heart_pred})
            else:
                heart_pred = 'we are sorry to tell you that you are more prone to heart disease'
                return render(request, 'positive.html', {'pred': heart_pred})
        else:
            return render(request, 'error.html', {'error': 'ML service error'})
    
    except Exception as e:
        return render(request, 'error.html', {'error': str(e)})

# Breast Cancer Prediction
def breast_cancer(request):
    return render(request, 'breast_cancer_pred.html')

def breast_cancer_result(request):
    try:
        # Collect all features from the request
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
        
        # Make API call to ML service
        response = requests.post(
            f"{settings.ML_SERVICE_URL}/predict", 
            json={
                'model_name': 'breast_cancer',
                'features': features
            }
        )
        
        if response.status_code == 200:
            cancer_pred = response.json()['prediction']
            
            if cancer_pred == 0:
                cancer_pred = 'we have predicted that the cancer is benign'
                return render(request, 'negative.html', {'pred': cancer_pred})
            else:
                cancer_pred = 'we have predicted that the cancer is malignant'
                return render(request, 'positive.html', {'pred': cancer_pred})
        else:
            return render(request, 'error.html', {'error': 'ML service error'})
    
    except Exception as e:
        return render(request, 'error.html', {'error': str(e)})

# Pneumonia Detection
def pneumonia(request):
    return render(request, 'pneumonia_pred.html')

from django.core.files.storage import FileSystemStorage
import base64

def pneumonia_result(request):
    try:
        fileobj = request.FILES['img']
        fs = FileSystemStorage()
        filePathName = fs.save(fileobj.name, fileobj)
        filePathName = fs.url(filePathName)
        
        # Read file and convert to base64
        with open('.' + filePathName, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Make API call to ML service
        response = requests.post(
            f"{settings.ML_SERVICE_URL}/predict", 
            json={
                'model_name': 'pneumonia',
                'image': encoded_image
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            pred = result['prediction']
            confidence = result['confidence']
            
            if pred == 1:
                return render(request, 'positive.html', {
                    'filePathName': filePathName, 
                    'pred': "you have pneumonia",
                    'percent': confidence,
                    'word1': "the model is ",
                    'word2': "% sure that it has detected pneumonia in the given xray "
                })
            else:
                return render(request, 'negative.html', {
                    'filePathName': filePathName,
                    'pred': "no pneumonia",
                    'percent': confidence,
                    'word1': "the model is ",
                    'word2': "% sure that the given xray is normal"
                })
        else:
            return render(request, 'error.html', {'error': 'ML service error'})
    
    except Exception as e:
        return render(request, 'error.html', {'error': str(e)})

# General Disease Prediction
def disease_pred(request):
    return render(request, 'disease_pred.html')

def disease_pred_result(request):
    try:
        # Collect symptoms
        symptoms = [request.GET['symptom1'], request.GET['symptom2'], 
                    request.GET['symptom3'], request.GET['symptom4'], 
                    request.GET['symptom5']]
        
        # Remove 'none' values
        list_updated = [symp for symp in symptoms if symp != 'none']
        
        if not list_updated:
            return render(request, 'error.html', {'error': 'Please enter at least one symptom'})
        
        # Make API call to ML service
        response = requests.post(
            f"{settings.ML_SERVICE_URL}/predict", 
            json={
                'model_name': 'general_disease',
                'symptoms': list_updated
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            disease = result['disease']
            doctor = result['recommended_doctor']
            
            return render(request, 'positive.html', {
                'pred': disease,
                'percent': doctor,
                'word1': "Please consult ",
                'suffer': "You might be suffering from "
            })
        else:
            return render(request, 'error.html', {'error': 'ML service error'})
    
    except Exception as e:
        return render(request, 'error.html', {'error': str(e)})
