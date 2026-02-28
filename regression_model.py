import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load the CSV file
df = pd.read_csv("day_5\dataset.csv")

# Features and target
X = df[["Experience_years"]]   # independent variable
y = df["Salary_lpa"]           # dependent variable

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LinearRegression()
model.fit(x_train, y_train)

# Coefficients
print("\nslope (m):", model.coef_[0])
print("intercept (b):", model.intercept_)

# Predictions
y_pred = model.predict(x_test)
for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# Prediction for new experience
new_exp = np.array([[9]])  # example: 9 years of experience
predicted_salary = model.predict(new_exp)
print(f"\nThe predicted salary for 9 years experience: {predicted_salary[0]:.2f} LPA")

# Visualization
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel("Experience (years)")
plt.ylabel("Salary (LPA)")
plt.title("Salary Prediction using Linear Regression")
plt.legend()
plt.show()