#!/usr/bin/env python
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,mean_squared_error
from sklearn.model_selection import cross_val_score,KFold
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

iris=load_boston()
X=iris.data
y=iris.target

tree1=KNeighborsRegressor()

tree1.fit(X,y)
p=tree1.predict(X)

print p
print(mean_squared_error(y,p))
plt.scatter(y,p,c='red')

tree2=DecisionTreeRegressor()
tree2.fit(X,y)
j=tree2.predict(X)
print j
print(mean_squared_error(y,j))
plt.scatter(y,j)
plt.show()

