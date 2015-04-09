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
data = np.load('dataout/dataPoints_a594.npy')
#data = np.load('dataout/dataPoints_cy5.npy')
#==============================================================================
# Threshold wrt to size of the feature
#==============================================================================
threshold = np.percentile(data[:,3],20)
rowboolean = data[:,3]<=threshold
data = np.compress(rowboolean, data, axis=0)
#==============================================================================
# Store all cases label in var cases
#==============================================================================
cases = np.unique(data[:,0])
#==============================================================================
# Selection of descriptors 
#==============================================================================
datacloud = data[:,8:]
#==============================================================================
# Partition data cloud using kmeans
#==============================================================================
datacloud = scale(datacloud)
n_samples, n_features = datacloud.shape
n_clusters = 20
km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=100, 
            tol=0.0001, precompute_distances='auto', verbose=0, 
            random_state=None, copy_x=True, n_jobs=-1)
km.fit(datacloud)    
centroids = km.cluster_centers_ #coordinates of centroids
labels = km.labels_ #labels of centroids
#==============================================================================
# normalize the centroids
#==============================================================================
#for ind in xrange(n_clusters):
#    centroids[ind,:] = np.true_divide(centroids[ind,:],np.linalg.norm(centroids[ind,:]))
#==============================================================================
# Subselect the most variable features and cluster on that, or simply do 
# hierarchical clustering on data
#==============================================================================
states = []
for case in cases:
    subset = data[:,0]==case
    histdata = labels[np.where(subset==True)[0]]
    id_counts = np.unique(histdata,return_counts=True)
    centroids_id = id_counts[0]
#==============================================================================
#     Prepare the state (pure state superposition)
#    TODO: check normalization
#==============================================================================
    centroids_counts = id_counts[1]
    state = np.zeros((1,np.shape(centroids)[1]))    
    for centroid in centroids_id:
        #Error: IndexError: index 19 is out of bounds for axis 0 with size 19!!!
        state = state + centroids_counts[centroid]*centroids[centroid,:]
    states.append(state)
#==============================================================================
# Plot the histrogram
#==============================================================================
    plt.title(str(case))
#    plt.ylim([0,centroids_counts.max()])
    plt.bar(xrange(len(centroids_counts)), centroids_counts)
    plt.draw()
    time.sleep(1.0)
    plt.clf()
#==============================================================================
# Save the states
#==============================================================================
np.savetxt('partitioning_data_20clusters_notnormalized.csv', np.vstack(pure_states), delimiter=',')
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
