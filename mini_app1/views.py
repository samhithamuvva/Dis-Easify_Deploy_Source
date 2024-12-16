import numpy as np
from django.shortcuts import render
import requests
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import base64
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)

def format_features(model_name, raw_features):
    """Convert features to appropriate type based on model"""
    try:
        if model_name == 'breast_cancer':
            return [float(x) for x in raw_features]  # 30 float features
        elif model_name == 'diabetes':
            return [int(x) for x in raw_features]    # 16 int features
        elif model_name in ['heart', 'heart_disease', 'heart1']:
            return [float(x) for x in raw_features]  # 13 float features
        elif model_name in ['disease_dtc', 'disease_gnb', 'general_disease']:
            return [int(x) for x in raw_features]    # int features
        return raw_features
    except ValueError as e:
        logger.error(f"Error converting features for {model_name}: {str(e)}")
        raise ValueError(f"Error converting features: {str(e)}")

# Home page
def home(request):
    return render(request, 'home.html')

# Diabetes Prediction
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
        
        response = requests.post(
            f"{settings.ML_SERVICE_URL}/predict", 
            json={
                'model_name': 'diabetes',
                'features': processed_features
            },
            timeout=10
        )
        
        if response.status_code == 200:
            dia_pred = response.json()['prediction'][0]
            
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
            logger.error(f"ML service error: {response.text}")
            return render(request, 'error.html', {'error': 'ML service error'})
    
    except Exception as e:
        logger.error(f"Diabetes prediction error: {str(e)}")
        return render(request, 'error.html', {'error': str(e)})

# Heart Disease Prediction
def heart_disease(request):
    return render(request, 'heart_css_pred.html')

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
        
        response = requests.post(
            f"{settings.ML_SERVICE_URL}/predict", 
            json={
                'model_name': 'heart',
                'features': processed_features
            },
            timeout=10
        )
        
        if response.status_code == 200:
            heart_pred = response.json()['prediction'][0]
            
            if heart_pred == 0:
                message = 'we are happy to tell you that you are less prone to heart disease'
                return render(request, 'negative.html', {'pred': message})
            else:
                message = 'we are sorry to tell you that you are more prone to heart disease'
                return render(request, 'positive.html', {'pred': message})
        else:
            logger.error(f"Heart disease ML service error: {response.text}")
            return render(request, 'error.html', {'error': 'ML service error'})
    
    except Exception as e:
        logger.error(f"Heart disease prediction error: {str(e)}")
        return render(request, 'error.html', {'error': str(e)})

# Breast Cancer Prediction
def breast_cancer(request):
    return render(request, 'breast_cancer_pred.html')

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
        
        response = requests.post(
            f"{settings.ML_SERVICE_URL}/predict", 
            json={
                'model_name': 'breast_cancer',
                'features': processed_features
            },
            timeout=10
        )
        
        if response.status_code == 200:
            cancer_pred = response.json()['prediction'][0]
            
            if cancer_pred == 0:
                message = 'we have predicted that the cancer is benign'
                return render(request, 'negative.html', {'pred': message})
            else:
                message = 'we have predicted that the cancer is malignant'
                return render(request, 'positive.html', {'pred': message})
        else:
            logger.error(f"Breast cancer ML service error: {response.text}")
            return render(request, 'error.html', {'error': 'ML service error'})
    
    except Exception as e:
        logger.error(f"Breast cancer prediction error: {str(e)}")
        return render(request, 'error.html', {'error': str(e)})
        
def pneumonia(request):
    return render(request, 'pneumonia_pred.html')

def pneumonia_result(request):
    if request.method != 'POST':
        return render(request, 'error.html', {'error': 'Invalid request method'})
        
    try:
        if 'img' not in request.FILES:
            return render(request, 'error.html', {'error': 'No image file uploaded'})
            
        fileobj = request.FILES['img']
        fs = FileSystemStorage()
        filePathName = fs.save(fileobj.name, fileobj)
        filepath = fs.path(filePathName)
        
        # Read file and convert to base64
        with open(filepath, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = requests.post(
            f"{settings.ML_SERVICE_URL}/predict_image", 
            json={
                'model_name': 'pneumonia',
                'image': encoded_image
            },
            timeout=30
        )
        
        # Clean up the file after processing
        fs.delete(filePathName)
        
        if response.status_code == 200:
            result = response.json()
            pred = result['prediction']
            confidence = result['confidence']
            
            context = {
                'pred': "you have pneumonia" if pred == 1 else "no pneumonia",
                'percent': confidence,
                'word1': "the model is ",
                'word2': "% sure that " + ("it has detected pneumonia" if pred == 1 else "the xray is normal")
            }
            
            template = 'positive.html' if pred == 1 else 'negative.html'
            return render(request, template, context)
        else:
            logger.error(f"Pneumonia ML service error: {response.text}")
            return render(request, 'error.html', {'error': 'ML service error'})
    
    except Exception as e:
        logger.error(f"Pneumonia prediction error: {str(e)}")
        return render(request, 'error.html', {'error': str(e)})
        
# General Disease Prediction
def disease_pred(request):
    return render(request, 'disease_pred.html')

def disease_pred_result(request):
    try:
        symptom1 = request.GET['symptom1']
        symptom2 = request.GET['symptom2']
        symptom3 = request.GET['symptom3']
        symptom4 = request.GET['symptom4']
        symptom5 = request.GET['symptom5']
        
        symp_list = [symptom1, symptom2, symptom3, symptom4, symptom5]
        list_updated = []

        for symp in symp_list:
            if symp != 'none':
                list_updated.append(symp)

        if list_updated == []:
            return render(request, 'error.html', {'error': 'Please enter at least one symptom'})
        
        # Your original symptom list
        symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
                   'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 
                   'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 
                   'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 
                   'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 
                   'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 
                   'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 
                   'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 
                   'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 
                   'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 
                   'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 
                   'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 
                   'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 
                   'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 
                   'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 
                   'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 
                   'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 
                   'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 
                   'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine', 
                   'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 
                   'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 
                   'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 
                   'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 
                   'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 
                   'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 
                   'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 
                   'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 
                   'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 
                   'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

        # Create feature vector
        index = [1 if symptom in list_updated else 0 for symptom in symptoms]
        
        # Make prediction using ML service
        response = requests.post(
            f"{settings.ML_SERVICE_URL}/predict",
            json={
                'model_name': 'disease_dtc',
                'features': index
            },
            timeout=10
        )
        
        if response.status_code == 200:
            prediction = response.json()['prediction'][0]
            
            # Read doctor recommendations from CSV
            doc = pd.read_csv('datasets/doctor_list.csv')
            c = doc.loc[prediction, "prognosis"]
            d = doc.loc[prediction, "Doctor"]
            
            hh = "Please consult "
            gg = "You might be suffering from "
            
            return render(request, 'positive.html', {
                'pred': c,
                'percent': d,
                'word1': hh,
                'suffer': gg
            })
            
        else:
            logger.error(f"Disease prediction error: {response.text}")
            return render(request, 'error.html', {'error': 'ML service error'})
            
    except Exception as e:
        logger.error(f"Disease prediction error: {str(e)}")
        return render(request, 'error.html', {'error': str(e)})
