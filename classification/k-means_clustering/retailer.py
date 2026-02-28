# Example: Customer Segmentation for a Retail Company 🛍️
# Business Context
# A retail chain wants to understand its customers better. Instead of treating everyone the same,
# they want to group customers into segments (like “budget shoppers,” “loyal premium buyers,” etc.)
#  so they can:
# - Personalize marketing campaigns
# - Recommend products more effectively
# - Improve customer retention

# Dataset (simplified)
# Imagine we have customer data with features like:
# - Annual Income (numeric)
# - Spending Score (numeric, based on purchase behavior)
# - Age (numeric)


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    'CustomerID': [1,2,3,4,5,6],
    'Age': [25,45,35,23,52,40],
    'AnnualIncome': [25000,60000,40000,20000,80000,50000],
    'SpendingScore': [30,70,50,20,90,60]
}

df=pd.DataFrame(data)
x=df[["Age","AnnualIncome","SpendingScore"]]
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

kmeans=KMeans(n_clusters=3,random_state=42)
df['cluster']=kmeans.fit_predict(x_scaled)
print(df)

plt.scatter(df['AnnualIncome'],df['SpendingScore'],c=df['cluster'],cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('customer segments')
plt.show()