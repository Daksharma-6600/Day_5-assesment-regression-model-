# Scenario 🎬
# A streaming platform wants to recommend movies to users based on their preferences.
#  Each movie is rated on three aspects:
# - Action Rating (how action‑packed it is)
# - Comedy Rating (how funny it is)
# - Drama Rating (how emotional it is)
# The platform collects data from past users about whether they liked (1) or didn’t like (0) certain movies.

# Question for Students
# Using the K‑Nearest Neighbors (KNN) algorithm:
# - Split the dataset into training and testing sets.
# - Scale the features (important for KNN).
# - Train models with different values of K (e.g., 1, 3, 5). Compare their accuracies.
# - Select the best model and predict whether a new user who prefers [Action=4, Comedy=2, Drama=4] will like the movie.
# - Discuss: How does changing K affect the model’s predictions?

# This scenario makes KNN relatable to recommendation systems like Netflix or Spotify, showing students how algorithms decide what they might enjoy.

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Dataset
x = [[5,2,3],[4,1,3],[5,5,4],[4,2,1],[5,1,5],
     [3,5,1],[1,4,3],[5,3,4],[2,1,4],[3,4,2]]
y = [1,1,0,0,1,0,0,1,1,0]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# Train KNN model (default k=5)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

# Evaluate accuracy
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Predict for a new user
new_user = scaler.transform([[4,2,4]])
print("Will they like it?", model.predict(new_user)[0])

