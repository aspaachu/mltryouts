import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('ex2.txt')
df.head()

# w/o scaling

X = df[['area', 'bedrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lr=LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(y_pred[:5])
print(y_test[:5])
print(mean_absolute_error(y_pred, y_test))
print(mean_squared_error(y_pred, y_test))

# w/ scaling

X = df[['area', 'bedrooms']]
y = df['price']

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lr=LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(y_pred[:5])
print(y_test[:5])
print(mean_absolute_error(y_pred, y_test))
print(mean_squared_error(y_pred, y_test))
