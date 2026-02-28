# Scenario 🏥
# A hospital wants to predict whether patients are at risk of developing diabetes based on their BMI (Body Mass Index). They collect data from 10 patients,
#  recording BMI values and whether the patient was diagnosed with diabetes (1 = diabetes, 0 = no diabetes).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('dtset.csv')
x=df[["BMI"]]
y=df[["Diabetes"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,prediction))

plt.scatter(x,y, color='red',label='data points')
x_range=np.linspace(15,55,200).reshape(-1,1)
y_prob=model.predict_proba(x_range)[:,1]
plt.plot(x_range,y_prob,color="red",label="Logistic curve")
plt.xlabel("bmi")
plt.ylabel("probablity of getting diabetes")
plt.legend()
plt.show()
