# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:47:14 2015

@author: silvano
"""
import cv2
import numpy as np
import skimage.io 
import os
#from modshogun import *

def Massage(tifffile):
    """
    Read the image file, projects it along z and optionally log 
    trasform and normalize
    """
    image_loc = skimage.io.imread(tifffile, plugin='tifffile')
    array_in = np.mean(image_loc, 0) # Project along z
    array_out = np.zeros(np.shape(array_in), dtype = np.uint8)    
    cv2.normalize(array_in,array_out,0,255,cv2.NORM_MINMAX,dtype=8)
    return array_out, array_in


def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """
    if os.path.getsize(filename) <= 0:
        return np.array([]), np.array([])
    f = np.load(filename)
    if f.size == 0:
        return np.array([]), np.array([])
    f = np.atleast_2d(f)
    return f[:,:7], f[:,7:] # feature locations, descriptors


def write_features_to_file(filename, locs, desc):
    np.save(filename, np.hstack((locs,desc)))    


def pack_keypoint(keypoints, descriptors):
    kpts = np.array([[kp.pt[0], kp.pt[1], kp.size,
                  kp.angle, kp.response, kp.octave,
                  kp.class_id]
                 for kp in keypoints])
    desc = np.array(descriptors)
    return kpts, desc


def unpack_keypoint(array):
    try:
        kpts = array[:,:7]
        desc = array[:,7:]
        keypoints = [cv2.KeyPoint(x, y, _size, _angle, _response, int(_octave), int(_class_id))
                 for x, y, _size, _angle, _response, _octave, _class_id in list(kpts)]
        return keypoints, np.array(desc)
    except(IndexError):
        return np.array([]), np.array([])


def process_image(imagename, resultname):
    img, original = Massage(imagename)
    sift = cv2.SIFT(nfeatures=0,nOctaveLayers=5,contrastThreshold=0.02,
                    edgeThreshold=10,sigma=0.5)
    k, des = sift.detectAndCompute(img,None)
    k, des = pack_keypoint(k, des) 
    write_features_to_file(resultname, k, des)
    

#==============================================================================
# From http://nbviewer.ipython.org/gist/kislayabhi/abb68be1b0be7148e7b7
# We take all the descriptors that we got from the three images above and find 
#similarity in between them. Here, similarity is decided by Euclidean distance 
#between the 128-element SIFT descriptors. Similar descriptors are clustered 
#into k number of groups. This can be done using Shogun's KMeans class. 
#These clusters are called bags of keypoints or visual words and they 
#collectively represent the vocabulary of the program. Each cluster has a 
#cluster center, which can be thought of as the representative descriptor of 
#all the descriptors belonging to that cluster. These cluster centers can be 
#found using the get_cluster_centers() method.    
#==============================================================================
def get_similar_descriptors(k, descriptor_mat):

    descriptor_mat=np.double(np.vstack(descriptor_mat))
    descriptor_mat=descriptor_mat.T

    #initialize KMeans in Shogun 
    sg_descriptor_mat_features=RealFeatures(descriptor_mat)

    #EuclideanDistance is used for the distance measurement.
    distance=EuclideanDistance(sg_descriptor_mat_features, sg_descriptor_mat_features)

    #group the descriptors into k clusters.
    kmeans=KMeans(k, distance)
    kmeans.train()

    #get the cluster centers.
    cluster_centers=(kmeans.get_cluster_centers())
    
    return cluster_centers
    
    