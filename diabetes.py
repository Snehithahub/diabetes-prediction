import pandas as pd

d=pd.read_csv(r"E:\Desktp_Backup\coding\archive\diabetes.csv")
d['Age']=pd.to_numeric(d['Age'], errors='coerce')
d['Age'].isna().sum()
#there is no null values to clean empty cells
#no wrong format
#EXPLORATORY DATA ANALYSIS
print(d.info())
print(d.describe())

'''
#VISUALIZATION
import seaborn as sns
import matplotlib.pyplot as plt
plt.subplot(5,3,1)
sns.histplot(d['Pregnancies'],bins=15,kde=False)
plt.title('Histogram with KDE - Pregnancies')
plt.subplot(5,3,2)
sns.histplot(d['Glucose'],bins=15,kde=False)
plt.title('GLUCOSE')
plt.subplot(5,3,3)
sns.histplot(d['BloodPressure'],bins=15,kde=False)
plt.title('BP')
plt.subplot(5,3,4)
sns.histplot(d['SkinThickness'],bins=15,kde=False)
plt.title('ST')
plt.subplot(5,3,5)
sns.histplot(d['Insulin'],bins=15,kde=False)
plt.title('INSULIN')
plt.subplot(5,3,6)
sns.histplot(d['BMI'],bins=15,kde=False)
plt.title('BMI')
plt.subplot(5,3,7)
sns.histplot(d['DiabetesPedigreeFunction'],bins=15,kde=False)
plt.title('DPF')
plt.subplot(5,3,8)
sns.histplot(d['Age'],bins=15,kde=False)
plt.title('AGE')

#CORRELATION
c=d.corr()

plt.subplot(5,3,9)
sns.boxplot(x='Outcome', y='Pregnancies', data=d)
plt.subplot(5,3,9)
sns.countplot(x='Glucose',data=d)
plt.subplot(5,3,10)
sns.heatmap(d.isnull())

#plt.figure(figsize=(9,9))
#sns.heatmap(c,annot=True,fmt=".2f")
#sns.pairplot(d,hue='Outcome', diag_kind='kde')

plt.show()
'''

#PREPROCESSING

#d.dropna(inplace=True)
#accuracy gor reduces cause of this

x=d.iloc[:,:8]
y=d.iloc[:,-1]

#TRAINING ,TESTING AND SPLITTING

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=32)

#CHOOSING OF MODEL
from sklearn.ensemble import RandomForestClassifier
l=RandomForestClassifier(n_estimators = 100,min_samples_split = 10,min_samples_leaf = 10).fit(x_train,y_train)
from sklearn import metrics
y_pred=l.predict(x_test)
print("accuracy:",metrics.accuracy_score(y_test,y_pred))
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

#TRAIN MODEL
import numpy as np
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

#SAVING AND CONTINUOUS REFINING
import joblib
joblib.dump(l,"diabetes_prediction.joblib")
load=joblib.load("diabetes_prediction.joblib")

s=np.array([1,80,110,20,15,19,20,20]).reshape(1,-1)
k=load.predict(s)
print(k)
result_label = "Yes" if k[0] == 1 else "No"
print("Prediction of Diabetes:", result_label)


s=np.array([5, 150, 80, 30, 100, 35, 45, 60]).reshape(1,-1)
k=load.predict(s)
print(k)
result_label = "Yes" if k[0] == 1 else "No"
print("Prediction of Diabetes:", result_label)




