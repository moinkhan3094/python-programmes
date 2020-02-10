# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 08:51:12 2020

@author: moin
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("survived.csv")
original_dataset=dataset.copy()

dataset.isnull()
dataset.isnull().sum()

dataset["Embarked"]=dataset["Embarked"].fillna(dataset["Embarked"].median())

plt.boxplot(dataset["Passengerid"])
plt.boxplot(dataset["Age"])
plt.boxplot(dataset["Fare"])
plt.boxplot(dataset["Sex"])
plt.boxplot(dataset["sibsp"])
plt.boxplot(dataset["zero"])
plt.boxplot(dataset["zero.1"])
plt.boxplot(dataset["zero.2"])
plt.boxplot(dataset["zero.3"])
plt.boxplot(dataset["zero.4"])
plt.boxplot(dataset["zero.5"])
plt.boxplot(dataset["zero.6"])
plt.boxplot(dataset["Parch"])
plt.boxplot(dataset["zero.7"])
plt.boxplot(dataset["zero.8"])
plt.boxplot(dataset["zero.9"])
plt.boxplot(dataset["zero.10"])
plt.boxplot(dataset["zero.11"])
plt.boxplot(dataset["zero.12"])
plt.boxplot(dataset["zero.13"])
plt.boxplot(dataset["zero.14"])
plt.boxplot(dataset["zero.16"])
plt.boxplot(dataset["Embarked"])
plt.boxplot(dataset["zero.17"])
plt.boxplot(dataset["zero.18"])
plt.boxplot(dataset["2urvived"])


outliers=["Age","Fare","sibsp","Parch"]
for col in outliers:
    percentiles=dataset[col].quantile([0.1,0.9]).values
    dataset[col]=dataset[col].clip(percentiles[0],percentiles[1])

plt.boxplot(dataset["Age"])
plt.boxplot(dataset["Fare"])
plt.boxplot(dataset["sibsp"])
plt.boxplot(dataset["Parch"])

percentiles=dataset["Fare"].quantile([0.01,0.09]).values
dataset["Fare"]=dataset["Fare"].clip(percentiles[0],percentiles[1])

plt.boxplot(dataset["Fare"])


percentiles=dataset["Parch"].quantile([0.01,0.09]).values
dataset["Parch"]=dataset["Parch"].clip(percentiles[0],percentiles[1])

plt.boxplot(dataset["Parch"])

del dataset["zero.1"]
del dataset["zero.2"]
del dataset["zero.3"]
del dataset["zero.4"]
del dataset["zero.5"]
del dataset["zero.6"]
del dataset["zero.7"]
del dataset["zero.8"]
del dataset["zero.9"]
del dataset["zero.10"]
del dataset["zero.11"]
del dataset["zero.12"]
del dataset["zero.13"]
del dataset["zero.14"]
del dataset["zero.15"]
del dataset["zero.16"]
del dataset["zero.17"]
del dataset["zero.18"]


del dataset["Passengerid"]
del dataset["Parch"]


X=dataset.iloc[:,0:6]
y=dataset["2urvived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from statsmodels.stats.outliers_influence import variance_inflation_factor
    
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns 

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


import statsmodels.api as sm
X_sm = sm.add_constant(X_train)
X_testsm = sm.add_constant(X_test)


from sklearn.metrics import r2_score
model_OHC = sm.OLS(y_train, X_sm).fit()
predictions = model_OHC.predict(X_testsm) 
r2_OHC=r2_score(y_test,y_pred)
model_OHC.summary()













