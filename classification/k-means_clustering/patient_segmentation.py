# Example: Hospital Patient Segmentation 🏥
# Business Context
# A hospital wants to improve patient care and resource allocation. Instead of treating all
# patients the same, they want to group them into segments based on health and lifestyle data.
#  This helps with:
# - Designing personalized treatment plans
# - Predicting high‑risk patients
# - Managing hospital resources more efficiently

# Dataset (simplified)
# Features we might use:
# - Age
# - BMI (Body Mass Index)
# - Number of yearly hospital visits
# - Chronic conditions count

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Step 1: Dataset
# -------------------------------
data = {
    'PatientID': [101,102,103,104,105,106],
    'Age': [25,60,45,30,70,50],
    'BMI': [22,30,28,24,35,27],
    'HospitalVisits': [1,5,3,2,7,4],
    'ChronicConditions': [0,2,1,0,3,1]
}

df = pd.DataFrame(data)

# -------------------------------
# Step 2: Feature Selection + Scaling
# -------------------------------
X = df[["Age","BMI","HospitalVisits","ChronicConditions"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 3: KMeans Clustering
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("Clustered DataFrame:")
print(df)

# -------------------------------
# Step 4: Visualization
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(df['Age'], df['HospitalVisits'], 
            c=df['Cluster'], cmap='viridis', s=100)

# Plot cluster centers (scaled back to original space)
centers = kmeans.cluster_centers_
centers_unscaled = scaler.inverse_transform(centers)
plt.scatter(centers_unscaled[:,0], centers_unscaled[:,2], 
            c='red', marker='X', s=200, label='Centers')

plt.xlabel('Age')
plt.ylabel('Hospital Visits')
plt.title('Hospital Patient Segments')
plt.legend()
plt.show()