# Hierarchircal Clustering

# Scenario Question 💼
# A retail bank wants to understand its customers better. They have
# collected data on Age and Annual Income for a sample of customers.
#  The goal is to group customers into meaningful segments so the bank can
#  design targeted loan offers, personalized investment plans, and marketing campaigns.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Dataset (Age, Income)
data = np.array([
    [25, 15000],
    [28, 16000],
    [30, 18000],
    [35, 22000],
    [40, 25000],
    [45, 60000],
    [50, 65000],
    [55, 70000]
])

# Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Agglomerative Clustering
hierarchical_cluster = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

cluster_labels = hierarchical_cluster.fit_predict(data_scaled)

print("Cluster labels:", cluster_labels)

# Create linkage matrix for dendrogram
linked = linkage(data_scaled, method='ward')

# Plot Dendrogram
plt.figure(figsize=(8,4))
dendrogram(linked,
           orientation="top",
           labels=range(1, len(data_scaled)+1),
           distance_sort="descending",
           show_leaf_counts=True)

plt.title("Customer Dendrogram (Age vs Income)")
plt.xlabel("Customer Index")
plt.ylabel("Distance")
plt.show()