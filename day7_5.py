#!/usr/bin/env python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

data1=pd.read_csv('banking.csv')
data=data1.as_matrix()


y=data[:,-1]
x=data[:,[0,10,11,12,13,15,16,17,18,19]]


#print(data.describe)
#print(data.shape)
#print(data.head)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

knn=KNeighborsClassifier()
knn.fit(x_train.astype(int),y_train.astype(int))
p=knn.predict(x_test)
print(p)

