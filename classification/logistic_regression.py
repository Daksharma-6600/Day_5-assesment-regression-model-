# Context: Predicting whether a driver files a claim based on age

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#create a simple dataset

x=np.array([[18], [20], [22], [25], [28],[30],[35], [40], [45],[50]])
y=np.array([1,1,1,0,0,0,0,0,1,1])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("accuracy :", accuracy_score(y_test,y_pred))

plt.scatter(x,y,color="blue",label="data points")
x_range=np.linspace(15,55,200).reshape(-1,1)
y_prob=model.predict_proba(x_range)[:,1]
plt.plot(x_range,y_prob,color="red",label="Logistic curve")
plt.xlabel("Driver age")
plt.ylabel("probablity of filling claims")
plt.legend()
plt.show()




