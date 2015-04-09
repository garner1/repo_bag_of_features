# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:46:58 2015

@author: silvano
"""

#==============================================================================
#Each case is a collection of features extracted using the SIFT descriptor 
#from different images. Here we use NMF to build the visual vocabulary.
# TODO: 
#   1)threshold wrt size of the feature
#   2)remove direction
#   3)consider x,y coordinates relative to image
#   4)consider the two channels, or dapi as well??
#==============================================================================
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import decomposition
from IPython import display
import pylab as pl
import pymf
import h5py
#==============================================================================
# Load data matrix [samples X features]
#features: case#,x,y,size,angle,response,octave,class_id
#==============================================================================
data = np.load('dataout/dataPoints_a594.npy')
#data = np.load('dataout/dataPoints_cy5.npy')
cases = np.unique(data[:,0])  #list all cases
#==============================================================================
# Selection of features over which perform the analysis
#==============================================================================
datacloud = data[:1000,8:]
#==============================================================================
# NMF
#==============================================================================
n_samples, n_features = datacloud.shape
n_components = 20
nmf_mdl = pymf.CHNMF(datacloud.T, num_bases=n_components, niter=2, show_progress=True)
nmf_mdl.initialization()
nmf_mdl.factorize()

#matrix_proj = 
#datacloud_proj = np.dot(datacloud,matrix_proj)
##==============================================================================
## Subselect the principal component features and cluster on them, or simply do 
## hierarchical clustering on data
##==============================================================================
#states = []
#for case in cases:
#    rowboolean = data[:,0]==case
#    subset = np.compress(rowboolean, datacloud_proj, axis=0)
#    state = np.sum(subset,axis=0)
#    state = np.true_divide(state, np.linalg.norm(state,ord=1))
#    states.append(state)
##==============================================================================
## Plot the histogram
##==============================================================================
#    plt.title(str(case))
##    plt.ylim([0,1])
#    plt.bar(xrange(len(state)), state)
#    plt.draw()
#    time.sleep(1.0)
#    plt.clf()
##==============================================================================
## Save the states
##==============================================================================
#np.savetxt('cy5_pca_data_50comp_normalized.csv', np.vstack(states), delimiter=',')
##==============================================================================
## Efficient selection of case 
##==============================================================================
##case = cases[0]
##rowboolean = data[:,0]==case
##case_des = np.compress(rowboolean, data, axis=0)
##case_des = case_des[:,8:]
##==============================================================================
##   Timing
##==============================================================================
##t= time.time()
##elapsed = time.time() - t
##print elapsed
##==============================================================================
## SVD 
##==============================================================================
##case_des = case_des - np.mean(case_des, axis=0)
##A=np.dot(case_des.T,case_des)
##U, S, V = np.linalg.svd(A)
##np.max(S)/np.min(S)
