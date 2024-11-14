import numpy as np
import pandas as pd

data = pd.read_csv("/Users/shazid/Desktop/Dis-Easify/Dis-Easify-main/datasets/diabetes.csv")

print(data)

print(data['class'].value_counts())

print(data.info)

print(data.isnull().sum())

print(data.nunique())

df = data.copy()

print(df)
df['Gender'] = df['Gender'].replace({'Female':0,'Male':1})
print(df['Gender'])

for column in df.columns.drop(['Age','Gender','class']):
  df[column] = df[column].replace({'No':0,'Yes':1})

df['class'] = df['class'].replace({'Positive':1,'Negative':0})

print(df)

x = df.drop('class',axis=True)
y = df['class']

print(x)
print(y)

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,shuffle=True,random_state=123)
print(x_train)
print(y_train)

print("--------random forest classifier---------")
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(x_train,y_train)
print("Train score   :",rfc.score(x_train,y_train)*100)
print("Test score   :",rfc.score(x_test,y_test)*100)
y_pred = rfc.predict(x_test)
print("predicted x_test")

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

pred = [39,0,1,1,1,0,1,0,0,1,0,1,1,0,0,0]
y_ext_rfc = rfc.predict([pred])


print(y_ext_rfc)
print("------------------------------------------")

print("-----decision tree classifier------")
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
print("Train score   :",dtc.score(x_train,y_train)*100)
print("Test score   :",dtc.score(x_test,y_test)*100)

dtc_pred = dtc.predict(x_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,dtc_pred))

pred = [39,0,1,1,1,0,1,0,0,1,0,1,1,0,0,0]
y_ext_dtc = dtc.predict([pred])
preed = dtc.predict_proba([pred])
print("predict proba is:")
print(preed)
print('score is ',preed.max()*100)

print(y_ext_dtc)

from sklearn.naive_bayes import MultinomialNB,GaussianNB
gausnb = GaussianNB()
gausnb.fit(x_train,y_train)
print("Train score   :",gausnb.score(x_train,y_train)*100)
print("Test score   :",gausnb.score(x_test,y_test)*100)

nb_pred = gausnb.predict(x_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,nb_pred))

pred = [39,0,1,1,1,0,1,0,0,1,0,1,1,0,0,0]
y_ext_nb = gausnb.predict([pred])
print(y_ext_nb)
from joblib import dump
dump(dtc, '../savedModels/diabetes_dtc_model.joblib')
#desision tree classifier has been uploaded using joblib

