# Scenario Question: Email Spam Filter
# A company has built a machine learning model to detect spam emails.
# - True labels (y_true): Whether each email was actually spam (1 = Spam, 0 = Not Spam).
# - Predictions (y_pred): What the model guessed.
# - Probabilities (y_prob): How confident the model was in each prediction.
# The company wants to evaluate the model using accuracy, precision, recall, F1 score, ROC-AUC,


from sklearn.metrics import(accuracy_score,precision_score,recall_score,
                             f1_score,confusion_matrix,classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Example true labels and model predictions
y_true = [1,0,1,1,0,1,0,0,1,0]
y_pred = [1,0,1,0,0,1,1,0,1,0]
y_prob = [0.9,0.1,0.8,0.4,0.2,0.85,0.6,0.15,0.7,0.3]

# Classification Report
print(classification_report(y_true, y_pred, target_names=["notspam","spam"]))

# Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",cmap="Purples", xticklabels=["notspam","spam"],
            yticklabels=["notspam","spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()