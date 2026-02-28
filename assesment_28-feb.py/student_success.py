
# Capstone Project: Student Success & Career Path Prediction

# Scenario

# The university wants to analyze student performance data to:c:\Users\Daksh Sharma\OneDrive\Desktop\Student Success & Career Path  - Sheet1.csv

# Predict exam scores (Regression).
# Classify students into “At Risk” vs. “On Track” categories (Classification).
# Cluster students into groups with similar study habits (Clustering).
# Recommend interventions (extra tutoring, workshops, counseling).

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, classification_report, accuracy_score

df = pd.read_csv("Student Success & Career Path  - Sheet1.csv")

df = df.drop(columns=['Student_ID'], errors='ignore')

if 'Gender' in df.columns:
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

df = df.dropna()

X_reg = df.drop(['Final_Exam_Score', 'Pass_Fail'], axis=1)
y_reg = df['Final_Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train)

y_pred_reg = reg_model.predict(X_test_scaled)

print("Regression Results ")
print("R2 Score:", r2_score(y_test, y_pred_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_reg)))


df['Risk_Status'] = df['Final_Exam_Score'].apply(lambda x: 1 if x >= 50 else 0)

X_clf = df.drop(['Final_Exam_Score', 'Pass_Fail', 'Risk_Status'], axis=1)
y_clf = df['Risk_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_model = LogisticRegression()
clf_model.fit(X_train_scaled, y_train)

y_pred_clf = clf_model.predict(X_test_scaled)

print("Classification Results")
print(classification_report(y_test, y_pred_clf))
print("Accuracy:", accuracy_score(y_test, y_pred_clf))


cluster_features = df[['Hours_Studied',
                       'Attendance (%)',
                       'Assignments_Submitted',
                       'Participation_Score',
                       'Previous_Sem_GPA']]

cluster_scaled = StandardScaler().fit_transform(cluster_features)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(cluster_scaled)

print("\nCluster Distribution:")
print(df['Cluster'].value_counts())