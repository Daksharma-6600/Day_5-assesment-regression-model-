from sklearn.tree import DecisionTreeClassifier ,export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x=[[720,60,1],[580,35,0],[700,55,1],[600,40,1],[750,80,1],[500,25,0],[680,50,1],[550,30,0],[730,70,1],[610,42,0]]
y=[1,0,1,0,1,0,1,0,1,0]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=42)
tree=DecisionTreeClassifier(max_depth=3,criterion='gini')
tree.fit(x_train,y_train)

feature_names=['credit_score','income','employed']
print(export_text(tree,feature_names=feature_names))

y_pred=tree.predict(x_test)
print("Accuracy:",y_pred,y_test)

applicant=[[690,52,1]]
decision=tree.predict(applicant)
print("decision :","approved" if decision[0]==1 else "failed")