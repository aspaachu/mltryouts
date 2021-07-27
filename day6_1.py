#!/usr/bin/env python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import cross_val_score,KFold

iris=load_iris()
X=iris.data
y=iris.target
kfold=KFold(n_splits=20,random_state=7)
tree1=DecisionTreeClassifier()
a=cross_val_score(tree1,X,y,cv=kfold)
tree1.fit(X,y)
p=tree1.predict(X)
print a
print(confusion_matrix(y,p))
print(classification_report(y,p))
print(accuracy_score(y,p))
