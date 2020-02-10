# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:40:40 2020

@author: moin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("vehicle_size.csv")
original_dataset=dataset.copy()

dataset.describe()
len(dataset)

dataset.isna()
dataset.notna()

dataset.isnull()
dataset.isnull().sum()

for col in dataset.columns:
    if dataset[col].dtypes == 'object' and dataset[col].isna().any() == True:
        print(col)
        
for col in dataset.columns:
    if dataset[col].dtypes == "float" and dataset[col].isna().any() == True:
        print(col)
        
mis_num=["Income","Monthly Premium Auto","Months Since Last Claim","Months Since Policy Inception","Number of Open Complaints","Number of Policies","Total Claim Amount"]        
for col in mis_num:
    dataset[col].fillna(dataset[col].median(),inplace=True)
    
mis_categorical=["EmploymentStatus","Gender","Location Code","Marital Status","Policy Type","Policy","Renew Offer Type","Sales Channel","Vehicle Class","Vehicle Size"]
for col in mis_categorical:
    dataset[col].fillna(dataset[col].mode()[0],inplace=True)
    
dataset.isnull().sum()


for col in dataset.columns:
    if dataset[col].dtypes == "float":
        print(col)
plt.boxplot(dataset["Customer Lifetime Value"])
plt.boxplot(dataset["Income"])
plt.boxplot(dataset["Monthly Premium Auto"])
plt.boxplot(dataset["Months Since Last Claim"])
plt.boxplot(dataset["Months Since Policy Inception"])
plt.boxplot(dataset["Number of Open Complaints"])
plt.boxplot(dataset["Number of Policies"])
plt.boxplot(dataset["Total Claim Amount"])


outliers=["Customer Lifetime Value","Monthly Premium Auto","Number of Open Complaints","Number of Policies","Total Claim Amount"]

for col in outliers:
    percentiles=dataset[col].quantile([0.1,0.9]).values
    dataset[col]=dataset[col].clip(percentiles[0],percentiles[1])
    
percentiles=dataset["Number of Open Complaints"].quantile([0.01,0.09]).values
dataset["Number of Open Complaints"]=dataset["Number of Open Complaints"].clip(percentiles[0],percentiles[1])
      
del dataset["Customer"]
del dataset["Gender"]
del dataset["Location Code"]
del dataset["Response"]

dfdummies_state=pd.get_dummies(dataset["State"],prefix="state")
dfdummies_coverage=pd.get_dummies(dataset["Coverage"],prefix="cover")
dfdummies_edu=pd.get_dummies(dataset["Education"],prefix="edu")
dfdummies_employee=pd.get_dummies(dataset["EmploymentStatus"],prefix="employee")
dfdummies_martial=pd.get_dummies(dataset["Marital Status"],prefix="martial")
dfdummies_ptype=pd.get_dummies(dataset["Policy Type"],prefix="ptype")
dfdummies_policy=pd.get_dummies(dataset["Policy"],prefix="policy")
dfdummies_renew=pd.get_dummies(dataset["Renew Offer Type"],prefix="renew")
dfdummies_sales=pd.get_dummies(dataset["Sales Channel"],prefix="sales")
dfdummies_vclass=pd.get_dummies(dataset["Vehicle Class"],prefix="vclass")
    
del  dfdummies_state["state_Arizona"]
del  dfdummies_coverage["cover_Basic"]
del  dfdummies_edu["edu_Bachelor"] 
del  dfdummies_employee["employee_Disabled"]
del  dfdummies_martial["martial_Divorced"]
del  dfdummies_ptype["ptype_Corporate Auto"]
del  dfdummies_policy["policy_Corporate L1"]
del  dfdummies_renew["renew_Offer1"]
del  dfdummies_sales["sales_Agent"]
del  dfdummies_vclass["vclass_Four-Door Car"]


dataset_2=pd.concat([dataset,dfdummies_state,dfdummies_coverage,dfdummies_edu,dfdummies_employee,dfdummies_martial,dfdummies_ptype,dfdummies_policy,dfdummies_renew,dfdummies_sales,dfdummies_vclass],axis=1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
dataset_2["Vehicle Size"] = labelencoder.fit_transform(dataset_2["Vehicle Size"])



y=dataset_2["Vehicle Size"]
del dataset_2["Vehicle Size"]
X=dataset_2.copy()

del X["Effective To Date"]
del X["State"]
del X["Coverage"]
del X["Education"]
del X["EmploymentStatus"]
del X["Marital Status"]
del X["Policy Type"]
del X["Policy"]
del X["Renew Offer Type"]
del X["Sales Channel"]
del X["Vehicle Class"]


from statsmodels.stats.outliers_influence import variance_inflation_factor
    
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns 
    
del X["Monthly Premium Auto"]




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

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



    
    
    
    