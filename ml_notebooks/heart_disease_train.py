import pandas as pd
import numpy as np

dataset = pd.read_csv("/Users/shazid/Desktop/Dis-Easify/Dis-Easify-main/datasets/heart.csv")

print(dataset)

print(dataset.info())

print(dataset.isnull().sum())

data = dataset.copy()
data.head()

print("x values")
x = data.drop(['target'],axis=1)
print(x)

print("y values")
y = data['target']
print(y)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler()
min_max = MinMaxScaler()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,shuffle=True,random_state=123)
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.fit_transform((x_test))
#x_train = min_max.fit_transform(x_train)
#x_test = min_max.fit_transform(x_test)
print(x_train)

print("random forest classifier")
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(x_train,y_train)
print('training score:',rfc.score(x_train,y_train)*100 )
print('test score:',rfc.score(x_test,y_test)*100 )
y_pred = rfc.predict(x_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

print("decision tree classifier")
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=2)
dtc.fit(x_train,y_train)
print("Train score   :",dtc.score(x_train,y_train)*100)
print("Test score   :",dtc.score(x_test,y_test)*100)
y_pred = dtc.predict(x_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

from joblib import dump
dump(rfc, '../savedModels/heart_rfc_model.joblib')
dump(dtc, '../savedModels/heart_dtc_model1.joblib')
#using random forest classifer for heart disease with training score = 99.8 and test_score = 95.33
#use standard scaler for heart disease prediction