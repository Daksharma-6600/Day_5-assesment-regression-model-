import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.DataFrame({
    "Exercise":[5,3,2,6,4,7,1,6,3,5],
    "Diet":[4,3,2,5,2,4,3,5,2,3],
    "Stress":[2,4,5,1,4,2,5,2,4,3],
    "AtRisk":[0,1,1,0,1,0,1,0,1,0]
})

X = data[["Exercise","Diet","Stress"]]
y = data["AtRisk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Try different values of K
for k in [1, 3, 5]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    print(f"K={k}  Accuracy={acc:.2f}")

# Best model (say K=3 performs best)
best_knn = KNeighborsClassifier(n_neighbors=3)
best_knn.fit(X_train, y_train)

# Predict for new user [Exercise=4, Diet=3, Stress=4]
new_user = scaler.transform([[4,5,1]])
print("Is the new user at risk?", best_knn.predict(new_user)[0])