# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:20:20 2020

@author: moin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


dataset=pd.read_csv("housingdata.csv")
original_data=dataset.copy()
dataset.columns=["p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","T"]

X=dataset.iloc[:,0:13]
y=dataset["T"]


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

del X["p6"]
del X["p11"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


def linear_reg(X, y, m, c, rounds, LR):
     L = len(y)
     for i in range(rounds):
          y_pred = (m * X) + c
          m -= LR*(-(2/L) * sum(X * (y - y_pred)))
          c -=LR*( -(2/L) * sum(y - y_pred))
          
     
     print(m,c)

import statsmodels.api as sm
X_sm = sm.add_constant(X_train)
X_testsm = sm.add_constant(X_test)



from sklearn.metrics import r2_score
model_OHC = sm.OLS(y_train, X_sm).fit()
predictions = model_OHC.predict(X_testsm) 
r2_OHC=r2_score(y_test,predictions)
model_OHC.summary()











