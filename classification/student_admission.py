# scenario: University Admissions Rulebook
# Imagine you’re an admissions officer at a university. Every day, students apply for admission, and you need to decide whether to accept or reject them.
# Instead of guessing, you build a rulebook (that’s your Decision Tree).

# 📋 The Data
# - Each applicant has:
# - High School GPA (how well they performed academically)
# - Entrance Exam Score (their standardized test performance)
# - Extracurriculars (1 = active in clubs/sports, 0 = not active)
# - Past applications are labeled:
# - 1 = Accepted
# - 0 = Rejected
# This past data is like your training experience.

# 👉 Just like the loan officer uses credit score, income, and employment status to decide, here the admissions officer uses GPA, exam scores, and extracurriculars to make decisions.
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 📂 Load dataset from CSV
data = pd.read_csv("University Dataset - Sheet1.csv")

# 🎯 Features (X) and Labels (y)
X = data[['HighSchool_GPA', 'Exam_Score', 'Extracurriculars']]
y = data['Admission_Label']

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🌳 Build Decision Tree
tree = DecisionTreeClassifier(max_depth=3, criterion='gini')
tree.fit(X_train, y_train)

# 📖 Show rules
feature_names = ['HighSchool_GPA', 'Exam_Score', 'Extracurriculars']
print(export_text(tree, feature_names=feature_names))

# 🎯 Evaluate accuracy
y_pred = tree.predict(X_test)
print("Predictions:", y_pred.tolist())
print("True Labels:", y_test.tolist())
print("Accuracy:", accuracy_score(y_test, y_pred))

# 🧑‍🎓 Test a new applicant
applicant = [[3.4, 1150, 1]]  # Example: GPA=3.4, Exam=1150, Extracurriculars=1
decision = tree.predict(applicant)
print("Decision:", "ACCEPTED ✅" if decision[0] == 1 else "REJECTED ❌")