import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv("titanic.csv")
df =df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df.head()

df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df.info()

df.isnull().sum()

mean_age = df['Age'].mean()
df['Age'].fillna(mean_age, inplace=True)

most_embarked = df['Embarked'].value_counts().idxmax()
df['Embarked'].fillna(most_embarked, inplace=True)

df['Embarked'].value_counts()

d1 = {'male': 1, 'female': 0}
d2 = {'S': 1, 'C': 2, 'Q': 3}

df['Sex'].replace(d1, inplace=True)
df['Embarked'].replace(d2, inplace=True)

df.head()

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

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
