import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('mining.csv')
df.head()

df.isnull().sum()

df['Formation'].value_counts()

df['Well Name'].value_counts()

dummies = pd.get_dummies(df['Formation'])
df = pd.concat([df, dummies], axis=1)
df.drop(['Formation'], inplace=True, axis=1)

dummies = pd.get_dummies(df['Well Name'])
df = pd.concat([df, dummies], axis=1)
df.drop(['Well Name'], inplace=True, axis=1)

df.head()

X = df.drop('Facies', axis=1)
y = df['Facies']

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(y_pred[:5])
print(y_test[:5])
print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
