import numpy as np
from django.shortcuts import render
from joblib import load
from sklearn.preprocessing import StandardScaler
import pandas as pd
scaler = StandardScaler()
#using standard scaler for heart disease prediction
#model = load('savedModels/model.joblib')
diabetes_model = load('savedModels/diabetes_dtc_model.joblib')
heart_disease_model = load('savedModels/heart_rfc_model.joblib')
heart_dtc = load('savedModels/heart_dtc_model1.joblib')
breast_cancer_model = load('savedModels/breast_cancer_rfc_model.joblib')
general_disease_model = load('savedModels/disease_gnb_model.joblib')




#for diabetes prediction we have used decision tree classifier model
#for heart disease prediction we have used random forest classifier
#for breast cancer we have used random forest classifier
#for general disease prediction we have used naive bayes (multinomialnb)
#for pneumonia detection we have used convolution neural networks(cnn)

# Create your views here.

#home page
def home(request):
    return render(request,'home.html')


#----------------------------------------------------------------------------------------------------------------

#diabetes prediction
def diabetes(request):
    return render(request,'diabetes.html')


def diabetes_result(request):
    age = request.GET['age']
    gender = request.GET['gender']
    polyuria = request.GET['polyuria']
    polydispia = request.GET['polydipsia']
    suddenWeightLoss = request.GET['suddenWeightLoss']
    Weakness = request.GET['Weakness']
    polyphagia = request.GET['polyphagia']
    genitalThrush = request.GET['genitalThrush']
    visualBlurring = request.GET['visualBlurring']
    itching = request.GET['itching']
    irritability = request.GET['irritability']
    delayedHealing = request.GET['delayedHealing']
    partialParesis = request.GET['partialParesis']
    muscleStiffness = request.GET['muscleStiffness']
    alopecia = request.GET['alopecia']
    obesity = request.GET['obesity']
    pred = [[age,gender,polyuria,polydispia,suddenWeightLoss,Weakness,polyphagia,genitalThrush,visualBlurring,itching,irritability,delayedHealing,partialParesis,muscleStiffness,alopecia,obesity]]
    print(pred)
    dia_pred = diabetes_model.predict(pred)
    print("printing the prediction")
    print(dia_pred)
    if dia_pred[0]==0:
        dia_pred="NEGATIVE"
        word1 = "We are happy to tell you that you have tested negative for diabetes"
        return render(request,'negative.html',{'pred':dia_pred})
    else:
        dia_pred="POSITIVE"
        word1 = "We are sorry to tell you that you have tested positive for diabetes"
        word2 = "Please consult a Endocrinologist"
        return render(request,'positive.html',{'pred':dia_pred,'word1':word1,'word2':word2})
    #return render(request,'diabetes_result.html',{'pred':dia_pred})
#---------------------------------------------------------------------------------------------------------------

#Heart disease prediction

def heart_disesase(request):
    return render(request,'heart_css_pred.html')

def heart_disease_result(request):
    age = request.GET['age']
    sex = request.GET['sex']#1 for male and 0 for female
    cp = request.GET['cp'] #chest pain type 1 or 2 or 3 or 4
    trestbps = request.GET['trestbps'] #resting blood pressure (in mm Hg on admission to the hospital) between 90 and 200
    chol = request.GET['chol'] #serum cholestoral in mg/dl between 126 and 564
    fbs = request.GET['fbs'] #(fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)
    restecg = request.GET['restecg'] # resting electrocardiographic results 0 or 1 or 2
    thalach = request.GET['thalach'] #maximum heart rate achieved 71 to 202
    exang = request.GET['exang'] #exercise induced angina (1 = yes; 0 = no)
    oldpeak = request.GET['oldpeak'] #ST depression induced by exercise relative to rest 0 to 6.2
    slope = request.GET['slope'] #the slope of the peak exercise ST segment 0 to 2
    ca = request.GET['ca'] #number of major vessels (0-3) colored by flourosopy
    thal = request.GET['thal'] #1 = normal; 2 = fixed defect; 3 = reversable defect
    heart_pred_list=[[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
    print(heart_pred_list)
    #heart_pred_scaled = scaler.fit_transform(heart_pred_list)
    heart_pred = heart_dtc.predict(heart_pred_list)

    print(heart_pred)
    if heart_pred[0]==0:
        heart_pred='we are happy to tell you that you are less prone to heart disease'
        return render(request, 'negative.html', {'pred': heart_pred})

    else:
        heart_pred='we are sorry to tell you that you are more prone to heart disease'
        return render(request,'positive.html',{'pred':heart_pred})

#----------------------------------------------------------------------------------------------------------------------------

#breast cancer prediction

def breast_cancer(request):
    return render(request,'breast_cancer_pred.html')

def breast_cancer_result(request):
    radiusMean = request.GET['radiusMean']
    textureMean = request.GET['textureMean']
    perimeterMean = request.GET['perimeterMean']
    areaMean = request.GET['areaMean']
    smoothnessMean = request.GET['smoothnessMean']
    compactnessMean = request.GET['compactnessMean']
    concavityMean = request.GET['concavityMean']
    concavePointsMean = request.GET['concavePointsMean']
    symmetryMean = request.GET['symmetryMean']
    fractalDimensionMean = request.GET['fractalDimensionMean']
    radiusSe = request.GET['radiusSe']
    textureSe = request.GET['textureSe']
    perimeterSe = request.GET['perimeterSe']
    areaSe = request.GET['areaSe']
    smoothnessSe = request.GET['smoothnessSe']
    compactnessSe = request.GET['compactnessSe']
    concavitySe = request.GET['concavitySe']
    concavePointsSe = request.GET['concavePointsSe']
    symmetrySe = request.GET['symmetrySe']
    fractalDimensionSe = request.GET['fractalDimensionSe']
    radiusWorst = request.GET['radiusWorst']
    textureWorst = request.GET['textureWorst']
    perimeterWorst = request.GET['perimeterWorst']
    areaWorst = request.GET['areaWorst']
    smoothnessWorst = request.GET['smoothnessWorst']
    compactnessWorst = request.GET['compactnessWorst']
    concavityWorst = request.GET['concavityWorst']
    concavePointsWorst = request.GET['concavePointsWorst']
    symmetryWorst = request.GET['symmetryWorst']
    fractalDimensionWorst = request.GET['fractalDimensionWorst']

    pred_list = [[radiusMean,textureMean,perimeterMean,areaMean,smoothnessMean,compactnessMean,concavityMean,concavePointsMean,symmetryMean,fractalDimensionMean,
             radiusSe,textureSe,perimeterSe,areaSe,smoothnessSe,compactnessSe,concavitySe,concavePointsSe,symmetrySe,fractalDimensionSe,
             radiusWorst,textureWorst,perimeterWorst,areaWorst,smoothnessWorst,compactnessWorst,concavityWorst,concavePointsWorst,symmetryWorst,fractalDimensionWorst]]
    print(pred_list)
    pred_scaled = scaler.fit_transform(pred_list)
    cancer_pred = breast_cancer_model.predict(pred_scaled)
    print(cancer_pred)
    #malignant is more dangerous than benign
    if cancer_pred[0]==0:
        cancer_pred='we have predicted that the cancer is benign'
        return render(request, 'negative.html', {'pred': cancer_pred})
    else:
        cancer_pred='we have predicted that the cancer is malignant'
        return render(request,'positive.html',{'pred':cancer_pred})


#---------------------------------------------------------------------------------------------------------------------------------------------

#pneumonia detection
def pneumonia(request):
    return render(request,'pneumonia_pred.html')

from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras_preprocessing import image
import tensorflow as tf
from tensorflow import Graph

model = load_model("savedModels/pneumonia2.h5")

img_height,img_width = 36,36

def pneumonia_result(request):
    fileobj = request.FILES['img']
    fs = FileSystemStorage()
    filePathName = fs.save(fileobj.name, fileobj)
    filePathName = fs.url(filePathName)
    testimage = '.'+filePathName
    img = image.load_img(testimage)
    #, target_size = (img_height, img_width)

    img = img.convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.reshape((1,36,36,1))
    img = img / 255.0
    pred = np.argmax(model.predict(img)[0])
    pred1 = np.argmax(model.predict(img))
    print("pred1 is")
    print(pred1)
    if pred1==0:

        h = model.predict(img)
        g = h[pred1][pred1]
        g= int(g*100)

        print(h)
    else:
        h = model.predict(img)
        g = h[0][pred1]
        g=int(g*100)
        print(g)
        print(h)

    print(pred)
    if pred ==1:
        print("you have pneumonia")
        pred="you have pneumonia"
        hh = "the model is "
        gg = "% sure that it has detected pneumonia in the given xray "

        return render(request, 'positive.html', {'filePathName': filePathName, 'pred': pred , 'percent':g,'word1':hh,'word2':gg})
    else:
        print("no pneumonia")
        pred="no pneumonia"
        hh = "the model is "
        gg = "% sure that the given xray is normal"
        return render(request,'negative.html',{'filePathName':filePathName,'pred':pred,'percent':g,'word1':hh,'word2':gg})
#----------------------------------------------------------------------------------------------------------------------------------------------

#general disease prediction

def disease_pred(request):
    return render(request,'disease_pred.html')

def disease_pred_result(request):
    symptom1 = request.GET['symptom1']
    symptom2 = request.GET['symptom2']
    symptom3 = request.GET['symptom3']
    symptom4 = request.GET['symptom4']
    symptom5 = request.GET['symptom5']
    print(symptom1,symptom2,symptom3,symptom4,symptom5)
    #print(symptom1)
    symp_list = [symptom1,symptom2,symptom3,symptom4,symptom5]
    list_updated=[]
    #below is another type for removing none from the list
    #for symp in list:
     #   if symp=='none':
      #      pass
       # else:
        #    list_updated.append(symp)
        #print(list_updated)
    if symp_list == ('none','none','none','none'):
        print("it is null")

    for symp in symp_list:
        if symp!='none':
            list_updated.append(symp)

    print("list updated")
    if list_updated== []:
        c = 'please enter alteast one symptom'
        print('empty')
    else:



        symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
                    'joint_pain',
                    'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
                    'spotting_ urination', 'fatigue',
                    'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
                    'lethargy', 'patches_in_throat',
                    'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
                    'dehydration', 'indigestion',
                    'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
                    'back_pain', 'constipation',
                    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
                    'fluid_overload',
                    'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
                    'throat_irritation',
                    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
                    'fast_heart_rate',
                    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
                    'dizziness', 'cramps',
                    'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
                    'enlarged_thyroid', 'brittle_nails',
                    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
                    'slurred_speech', 'knee_pain', 'hip_joint_pain',
                    'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements',
                    'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
                    'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine',
                    'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
                    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
                    'abnormal_menstruation', 'dischromic _patches',
                    'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
                    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
                    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
                    'distention_of_abdomen', 'history_of_alcohol_consumption',
                    'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking',
                    'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
                    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
                    'yellow_crust_ooze']

        diseases = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
                    'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
                    ' Migraine', 'Cervical spondylosis',
                    'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid',
                    'hepatitis A',
                    'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
                    'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
                    'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
                    'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
                    'Impetigo']
        index = []
        for x in range(0, len(symptoms)):
            index.append(0)
        for k in range(0,len(symptoms)):
            for z in list_updated:
                if(z==symptoms[k]):
                    index[k]=1

        doc = pd.read_csv('datasets/doctor_list.csv')
        prog = doc["prognosis"]
        b = doc["Doctor"]

        print('index:')

        print(index)
        inputtest = [index]
        print('inputtest is:')
        print(inputtest)
        predict = general_disease_model.predict(inputtest)
        z = predict[0]
        c = diseases[z]
        d = b[z]
        print("please consult ",d)
        print(c)
        print(d)

        hh = "Please consult  "
        gg = "You might me suffering from "



    return render(request,'positive.html',{'pred':c,'percent':d,'word1':hh,'suffer':gg})
