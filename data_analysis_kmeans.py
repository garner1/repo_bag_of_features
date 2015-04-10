# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 08:16:26 2015

@author: silvano
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
#==============================================================================
# Load data matrix [samples X features]
#features: case#,x,y,size,angle,response,octave,class_id
#==============================================================================
#data = np.load('dataout/filtered_a594.npy')
data = np.load('dataout/filtered_cy5.npy')
#==============================================================================
# Store all cases label in var cases
#==============================================================================
cases = np.unique(data[:,0])
#==============================================================================
# Selection of descriptors 
#==============================================================================
datacloud = data[:,8:]
#==============================================================================
# Partition data cloud using kmeans.
# kmeans works well with euclidean distances, not others, (not good for hist).
# It is good to scale data before in order not to bias on specif features.
#==============================================================================
datacloud = scale(datacloud)
n_samples, n_features = datacloud.shape
n_clusters = 20
km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=100, 
            tol=0.0001, precompute_distances='auto', verbose=0, 
            random_state=None, copy_x=True, n_jobs=-1)
km.fit(datacloud)    
centroids = km.cluster_centers_ #coordinates of centroids [numb_c X 128]
labels = km.labels_ #labels of centroids
#==============================================================================
# Subselect the most variable features and cluster on that, or simply do 
# hierarchical clustering on data
#==============================================================================
states = []
for case in cases:
#    case = cases[1]
    subset = data[:,0]==case
    histdata = labels[np.where(subset==True)[0]] #select all the cluster labels for the case
    id_counts = np.unique(histdata,return_counts=True) #id_counts is a 2 element list
    centroids_id = id_counts[0]
    centroids_counts = id_counts[1]
    states.append(centroids_counts)
#==============================================================================
# Plot the histrogram
#==============================================================================
    plt.title(str(int(case)))
#    plt.ylim([0,centroids_counts.max()])
    plt.bar(xrange(len(centroids_counts)), centroids_counts)
    plt.draw()
    time.sleep(1.0)
    plt.clf()
#==============================================================================
# Save the states
#==============================================================================
#np.save('./dataout/centroid_coord_a594',centroids)        
#np.savetxt('dataout/bof_data_20clusters_a594.csv', np.vstack(states), delimiter=',')
np.save('./dataout/centroid_coord_cy5',centroids)        
np.savetxt('dataout/bof_data_20clusters_cy5.csv', np.vstack(states), delimiter=',')
#==============================================================================
# Efficient selection of case 
#==============================================================================
#case = cases[0]
#rowboolean = data[:,0]==case
#case_des = np.compress(rowboolean, data, axis=0)
#case_des = case_des[:,8:]
#==============================================================================
#   Timing
#==============================================================================
#t= time.time()
#elapsed = time.time() - t
#print elapsed
#==============================================================================
# SVD 
#==============================================================================
#case_des = case_des - np.mean(case_des, axis=0)
#A=np.dot(case_des.T,case_des)
#U, S, V = np.linalg.svd(A)
#np.max(S)/np.min(S)
