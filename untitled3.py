# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:15:27 2020

@author: moin
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


dataset=pd.read_csv("C:/Users/moin/Desktop/OPERATIONS/mavoix/data_science_sample_data_v2.csv")
original_dataset=dataset.copy()

del dataset["Institute"]
del dataset["Current City"]
del dataset["Performance_12"]
del dataset["Performance_10"]


dataset.isnull().sum()
dataset["Other skills"]=dataset["Other skills"].fillna(0)
dataset["Degree"]=dataset["Degree"].fillna(0)
dataset["Stream"]=dataset["Stream"].fillna(0)
dataset["Current Year Of Graduation"]=dataset["Current Year Of Graduation"].fillna(0)
dataset["Performance_PG"]=dataset["Performance_PG"].fillna(0)
dataset["Performance_UG"]=dataset["Performance_UG"].fillna(0)

dataset.isnull().sum()

dfdummies_state=pd.get_dummies(dataset["Degree"],prefix="Degree")
dfdummies_Otherskills=pd.get_dummies(dataset["Other skills"],prefix="Other skills")
dfdummies_stream=pd.get_dummies(dataset["Stream"],prefix="Stream")
dfdummies_graduation=pd.get_dummies(dataset["Current Year Of Graduation"],prefix="Current Year Of Graduation")

del dataset["Degree"]
del dataset["Other skills"]
del dataset["Stream"]
del dataset["Current Year Of Graduation"]


dataset['Performance_UG'] = dataset['Performance_UG'].str.split('/').str[0].astype('float64') / dataset['Performance_UG'].str.split('/').str[1].astype('float64')
dataset['Performance_PG'] = dataset['Performance_PG'].str.split('/').str[0].astype('float64') / dataset['Performance_PG'].str.split('/').str[1].astype('float64')


dataset_2=pd.concat([dataset,dfdummies_state,dfdummies_Otherskills,dfdummies_stream,dfdummies_graduation],axis=1)

dataset_2['Performance_UG'] = dataset_2['Performance_UG'].replace(np.nan, 0)
dataset_2['Performance_PG'] = dataset_2['Performance_PG'].replace(np.nan, 0)

del dataset_2["Application_ID"]

X=dataset_2.iloc[:,:].values



from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('data_scientist_v2')
plt.xlabel('Applications')
plt.ylabel('Selected applicants')
plt.legend()
plt.show()










