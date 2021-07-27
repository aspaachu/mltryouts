#!/usr/bin/env python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np

data1=pd.read_csv('banking.csv')
data=data1.as_matrix()


y=data[:,-1]

le=preprocessing.LabelEncoder()
le.fit(data[:,1])
k=le.transform(data[:,1])
print('Lables: ',data[:,1])
print('Encoded Lables: ',k)


x=data[:,[0,10,11,12,13,15,16,17,18,19]]

#print(len(data[:,-1]))
#print(len(k))

print('Initial Shape: ',x.shape)

x=np.column_stack((x,k))

print('Shape After: ',x.shape)
#print(x.shape)
#print(type(x))

#print(data.describe)
#print(data.shape)
#print(data.head)

knn=KNeighborsClassifier()
knn.fit(x.astype(int),y.astype(int))
p=knn.predict(x)
print(p)

