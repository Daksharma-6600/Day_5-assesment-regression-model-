import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Features: [word_count, has_link, caps_ratio]
x= np.array([
    [50, 1, 0.8],   # SPAM
    [200, 0, 0.1],  # Not spam
    [30, 1, 0.9],   # SPAM
    [180, 0, 0.05], # Not spam
    [10, 1, 0.95],  # SPAM
    [220, 0, 0.08], # Not spam
])
y = np.array([1, 0, 1, 0, 1, 0])  # 1=Spam, 0=Not spam])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_prediction=model.predict(x_test)
proba=model.predict_proba(x_test)[:,1]

print("prediction",y_prediction)
print("probablity",proba.round(2))
print("Accuracy",accuracy_score(y_prediction,y_test))

new_email=[[15,1,0.88]]
print("new email is spam?",model.predict(new_email)[0])