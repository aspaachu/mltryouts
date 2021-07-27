#!/usr/bin/env python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data1=pd.read_csv('pimaindians.csv')
data=data1.as_matrix()

y=data[:,8]
x=data[:,[0,1,2,3,4,5,6,7]]

#print(data.describe)
#print(data.shape)
#print(data.head)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

rf=RandomForestClassifier()
rf.fit(x_train,y_train)
p=rf.predict(x_test)
print(accuracy_score(y_test,p))

