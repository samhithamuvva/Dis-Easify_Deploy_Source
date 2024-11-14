from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('diabetes', views.diabetes, name='diabetes'),
    path('heart_disease', views.heart_disesase, name='heart_disease'),
    path('diabetes_result', views.diabetes_result, name='diabetes_result'),
    path('heart_disease_result', views.heart_disease_result, name='heart_disease_result'),
    path('breast_cancer', views.breast_cancer, name='breast_cancer'),
    path('breast_cancer_result', views.breast_cancer_result, name='breast_cancer_result'),
    path('pneumonia', views.pneumonia, name='pneumonia_detection'),
    path('pneumonia_result', views.pneumonia_result, name='pneumonia_result'),
    path('disease_pred', views.disease_pred, name='disease_pred'),
    path('disease_pred_result', views.disease_pred_result, name='disease_result'),
    #path('result', views.form_result, name='result'),
]