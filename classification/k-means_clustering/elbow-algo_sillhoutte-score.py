# Scenario Question 💼
# A telecommunications company has collected data on 500 customers, including their monthly bill
# amount, average call duration, internet usage, and number of support calls. The company wants
# to group customers into meaningful segments to design targeted marketing campaigns and improve customer
#  service.
# You are tasked with:
# - Using K‑Means clustering to explore possible customer segments.
# - Applying the Elbow Method to determine where adding more clusters stops giving significant improvement.
# - Using the Silhouette Score to validate which number of clusters produces the most well‑separated
# and meaningful groups.


# Scenario Question 💼
# A telecommunications company has collected data on 500 customers, including their monthly bill amount, average call duration, internet usage, and number of support calls. The company wants to group customers into meaningful segments to design targeted marketing campaigns and improve customer service.
# You are tasked with:
# - Using K‑Means clustering to explore possible customer segments.
# - Applying the Elbow Method to determine where adding more clusters stops giving significant improvement.
# - Using the Silhouette Score to validate which number of clusters produces the most well‑separated and meaningful groups

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -------------------------------
# Step 1: Create synthetic dataset
# -------------------------------
data = {
    'CustomerID': range(1, 501),
    'MonthlyBill': np.random.randint(20, 200, 500),       # monthly bill in $
    'CallDuration': np.random.randint(50, 500, 500),      # avg monthly call minutes
    'InternetUsage': np.random.randint(10, 300, 500),     # GB per month
    'SupportCalls': np.random.randint(0, 10, 500)         # number of support calls
}
df = pd.DataFrame(data)

# -------------------------------
# Step 2: Feature Selection + Scaling
# -------------------------------
X = df[["MonthlyBill","CallDuration","InternetUsage","SupportCalls"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 3: Elbow Method
# -------------------------------
inertia = []
K_range = range(2, 11)  # test clusters from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method for Optimal k")
plt.show()

# -------------------------------
# Step 4: Silhouette Score Validation
# -------------------------------
silhouette_scores = {}
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores[k] = score

print("Silhouette Scores by k:")
for k, score in silhouette_scores.items():
    print(f"k={k}: {score:.3f}")

# -------------------------------
# Step 5: Final Clustering (choose best k)
# -------------------------------
best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nBest number of clusters based on silhouette score: {best_k}")

final_kmeans = KMeans(n_clusters=best_k, random_state=42)
df['Cluster'] = final_kmeans.fit_predict(X_scaled)

print("\nClustered DataFrame (first 10 rows):")
print(df.head(10))

# -------------------------------
# Step 6: Visualization (MonthlyBill vs InternetUsage)
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(df['MonthlyBill'], df['InternetUsage'], 
            c=df['Cluster'], cmap='viridis', s=50)

centers = final_kmeans.cluster_centers_
centers_unscaled = scaler.inverse_transform(centers)
plt.scatter(centers_unscaled[:,0], centers_unscaled[:,2], 
            c='red', marker='X', s=200, label='Centers')

plt.xlabel('Monthly Bill ($)')
plt.ylabel('Internet Usage (GB)')
plt.title('Telecom Customer Segments')
plt.legend()
plt.show()