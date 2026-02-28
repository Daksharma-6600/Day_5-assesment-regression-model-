# Scenario Question: Predicting Titanic Survival
# Researchers are studying the Titanic disaster and want to build models that predict whether a
#  passenger would survive or not survive based on their information.
# - Features used:
# - Passenger class (pclass)
# - Gender (sex)
# - Age (age)
# - Number of siblings/spouses aboard (sibsp)
# - Number of parents/children aboard (parch)
# - Ticket fare (fare)
# - Label:
# - 1 = Survived
# - 0 = Died
# The researchers train three different models:
# - Logistic Regression
# - K-Nearest Neighbors (KNN) with k=5
# - Decision Tree with max depth = 4
# They then evaluate each model using a classification report (precision, recall, F1-score, accuracy).

# ❓ Questions for Learners
# - Which model performs best at predicting survival, and why?
# - How does Logistic Regression differ from Decision Tree in terms of interpretability?
# # - Why is scaling applied before training Logistic Regression and KNN, but not strictly needed
#  for Decision Trees?
# - Looking at the classification report, what do precision and recall mean in the context of survival
#  predictions?
# - Precision → Of those predicted to survive, how many actually survived?
# - Recall → Of all who truly survived, how many were correctly predicted?
# - If you were a historian, which model would you trust more to explain survival patterns, and why?

# Use the below pre-loaded dataset:

# 1. Load data (use seaborn's built-in dataset)

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = sns.load_dataset('titanic')

df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'survived']].copy()

df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

df = df.dropna()

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test_scaled)
knn_pred = knn_model.predict(X_test_scaled)
tree_pred = tree_model.predict(X_test)

print("===== Logistic Regression =====")
print(classification_report(y_test, log_pred))
print("Accuracy:", accuracy_score(y_test, log_pred))

print("\n===== KNN (k=5) =====")
print(classification_report(y_test, knn_pred))
print("Accuracy:", accuracy_score(y_test, knn_pred))

print("\n===== Decision Tree (max_depth=4) =====")
print(classification_report(y_test, tree_pred))
print("Accuracy:", accuracy_score(y_test, tree_pred))

results = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN (k=5)", "Decision Tree (depth=4)"],
    "Accuracy": [
        accuracy_score(y_test, log_pred),
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, tree_pred)
    ]
})

print("\nModel Comparison:")
print(results.sort_values(by="Accuracy", ascending=False))

cm_log = confusion_matrix(y_test, log_pred)
plt.figure()
plt.imshow(cm_log)
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ["Died", "Survived"])
plt.yticks([0,1], ["Died", "Survived"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_log[i, j])
plt.show()

cm_knn = confusion_matrix(y_test, knn_pred)
plt.figure()
plt.imshow(cm_knn)
plt.title("KNN (k=5) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ["Died", "Survived"])
plt.yticks([0,1], ["Died", "Survived"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_knn[i, j])
plt.show()

cm_tree = confusion_matrix(y_test, tree_pred)
plt.figure()
plt.imshow(cm_tree)
plt.title("Decision Tree (max_depth=4) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ["Died", "Survived"])
plt.yticks([0,1], ["Died", "Survived"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_tree[i, j])
plt.show()