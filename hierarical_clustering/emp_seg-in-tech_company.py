# Scenario: Employee Segmentation in a Tech Company 💻
# Business Problem
# A tech company wants to understand its employees better to design training programs
# and salary structures. They collected data on each employee’s Age and Annual Salary.
# Management believes employees can be grouped into clusters such as:
# - Young, entry‑level employees
# - Mid‑career professionals
# - Senior, high‑earning employees
# They decide to use hierarchical clustering to explore these segments.


# data = np.array([
#     [25, 15000],
#     [28, 16000],
#     [30, 18000],
#     [35, 22000],
#     [40, 25000],
#     [45, 60000],
#     [50, 65000],
#     [55, 70000]
# ])


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

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

scalar=StandardScaler()
data_scaled=scalar.fit_transform(data)

hierarchical_cluster = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)
cluster_labels=hierarchical_cluster.fit_predict(data_scaled)
print("Cluster labels:", cluster_labels)

linked = linkage(data_scaled, method='ward')

# Plot Dendrogram
plt.figure(figsize=(8,4))
dendrogram(linked,
           orientation="top",
           labels=range(1, len(data_scaled)+1),
           distance_sort="descending",
           show_leaf_counts=True)

plt.title("Employee Dendrogram (Age vs AnnualSalary)")
plt.xlabel("Employee Index")
plt.ylabel("Distance")
plt.show()