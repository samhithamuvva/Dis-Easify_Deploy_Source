import pandas as pd
import numpy as np

dataset = pd.read_csv("/Users/shazid/Desktop/Dis-Easify/Dis-Easify-main/datasets/breast_cancer_data.csv")

print(dataset)

print(dataset.info())

print(dataset.isnull().sum())

data = dataset.copy()

print(data)

data = data.drop(['Unnamed: 32'],axis=1)
data = data.drop(['id'],axis=1)

data['diagnosis'] = data['diagnosis'].replace({'M':1,'B':0})
#B,12.98,19.35,84.52,514,0.09579,0.1125,0.07107,0.0295,0.1761,0.0654,0.2684,0.5664,2.465,20.65,0.005727,0.03255,0.04393,0.009811,0.02751,0.004572,14.42,21.95,99.21,634.3,0.1288,0.3253,0.3439,0.09858,0.3596,0.09166

x = data.drop("diagnosis",axis=1)
y = data['diagnosis']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,shuffle=True,random_state=123)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
print("Train score   :",dtc.score(x_train,y_train)*100)
print("Test score   :",dtc.score(x_test,y_test)*100)
y_pred = dtc.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

print("random forest classifier")
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(x_train,y_train)
print("Train score   :",rfc.score(x_train,y_train)*100)
print("Test score   :",rfc.score(x_test,y_test)*100)
y_pred = rfc.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

from joblib import dump
dump(rfc, '../savedModels/breast_cancer_rfc_model.joblib')

