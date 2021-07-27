import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('ex3.txt')
df.head()

sns.scatterplot(x='score1', y='score2', hue='admission', data=df )

X = df[['score1', 'score2']]
y = df['admission']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
#print X_test
y_pred = knn.predict(X_test)

print(y_pred[:5])
print(y_test[:5])
print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
