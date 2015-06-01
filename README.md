BAG-OF-FEATURES FOR TRANSCRIPTOMIC IMAGE-CLASSIFICATION
"""
Created on Thu Apr  9 16:45:36 2015

@author: silvano
"""
The idea is to use a bag-of-features approach to classify cases. 

Pipeline:
1) extract_SIFT_descriptor.py: Extract SIFT descriptors 
2) filter_and_see_features.py: Filter descriptors wrt size and removes inter-channel 
   overlapping keypoints 
3) data_analysis_{kmeans,minibatchkmeans,NMF,PCA}.py: 
   partition data according to some algo. At the moment only kmeans, minibatchkmeans, 
   PCA work well. 
4) classification.R: classify cases (find the right distance 
   measure btw histograms, see scipy.distance)
