# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:24:37 2015

@author: silvano
"""
#==============================================================================
#Each case is a collection of features extracted using the SIFT descriptor 
#from different images. Here we use PCA to build the visual vocabulary.
# TODO: 
#   1)threshold wrt size of the feature
#   2)remove direction
#   3)consider x,y coordinates relative to image
#   4)consider the two channels, or dapi as well??
#==============================================================================
import numpy as np
import time
import matplotlib.pyplot as plt
#from sklearn.preprocessing import scale
#from IPython import display
#import pylab as pl
#==============================================================================
# Load data matrix [samples X features]
#features: case#,x,y,size,angle,response,octave,class_id
#==============================================================================
#data = np.load('dataout/dataPoints_a594.npy')
data = np.load('dataout/dataPoints_cy5.npy')
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
# PCA
#==============================================================================
n_samples, n_features = datacloud.shape
n_components = 20
cov_mat = np.cov(datacloud.T)
eig_val, eig_vec = np.linalg.eig(cov_mat)
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()
# Select top 20
matrix_proj = np.hstack((eig_pairs[i][1].reshape(n_features,1)) for i in range(n_components))
#==============================================================================
# Low dimensional projection
#==============================================================================
datacloud_proj = np.dot(datacloud,matrix_proj)
#==============================================================================
# For each case, take the mean of each component in the low-dim space
#==============================================================================
states = []
for case in cases:
    rowboolean = data[:,0]==case
    subset = np.compress(rowboolean, datacloud_proj, axis=0)
    state = np.mean(subset,axis=0)
    fluctuation = np.std(subset,axis=0)
#    state = np.true_divide(state, np.linalg.norm(state,ord=1))
    states.append(state)
#==============================================================================
# Plot the histogram
#==============================================================================
    plt.title(str(case))
#    plt.ylim([-150,150])
    plt.bar(xrange(len(state)),state,yerr=fluctuation)
    plt.draw()
    time.sleep(1.0)
    plt.clf()
#==============================================================================
# Save the states
#==============================================================================
np.savetxt('cy5_pca_c20_t10.csv', np.vstack(states), delimiter=',')
#np.savetxt('a594_pca_c20_t10.csv', np.vstack(states), delimiter=',')
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
