# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:18:54 2015

@author: silvano
"""
import os
from module import Massage, pack_keypoint, write_features_to_file
import subprocess 
import numpy as np
import cv2
import itertools
#import time
#import matplotlib.pyplot as plt
#import skimage

#==============================================================================
# Check if output data file already exists
#==============================================================================
if os.path.exists('./dataout/dataPoints.npy'): 
    answer = raw_input("Do you want to cancel the old data file (if no it will be saved to ./dataout/dataPoints.npy.old)?[y or n] ")
    if answer=='y': 
        subprocess.call(["rm","./dataout/dataPoints.npy"])
    else:
        subprocess.call(["mv","./dataout/dataPoints.npy","./dataout/dataPoints.npy.old"])        
#==============================================================================
# Location of the tif files
#==============================================================================
dir_name = '/home/garner1/Work/BreastCancerAnalysis/data_input/Immagini_Casi_smFISH_Laura'
#==============================================================================
# Open the preselected list of valid files, to avoid bad tif files
#==============================================================================
f = open('./datain/mat_files_list.txt','r')
#==============================================================================
# for each line in the list of valid files
#==============================================================================
ii = 0 #counts iteration in the loop
#numb_of_tiff_images = 32
#for line in itertools.islice(f, numb_of_tiff_images): #to parse a limited numb of images
for line in f:    #to parse all images
    ii += 1
    print ii, ' of 2977'
#==============================================================================
#     parse the path to the tif file
#==============================================================================
#    filename = f.readline()
    filename = line
    case = filename.split('/')[1]
    number = filename.split('/')[2].split('_')[1].split('.')[0] 
    tag = np.uint(case.split('_')[2])
    filename_prefix1 = "a594"    
    filename_prefix2 = "cy5"
    filename_suffix = "tif"
    a594 = os.path.join(dir_name, case, filename_prefix1 + "_" + number + "." 
                    + filename_suffix)
    cy5 = os.path.join(dir_name, case, filename_prefix2 + "_" + number + "." 
                    + filename_suffix)
    imagefile = cy5
#==============================================================================
#   Preprocess the file: 
#   image: 8bit image projected along z
#   original: 16bit projected along z image
#==============================================================================
    img, original = Massage(imagefile)
#==============================================================================
#   SIFT descriptor: good enough at point detection
#   these are good parameters: nfeatures=0,nOctaveLayers=5,
#                    contrastThreshold=0.02,
#                    edgeThreshold=10,sigma=0.5
#   kp[ind].pt or .size, .angle, .response, .octave, .class_id
#==============================================================================
    sift = cv2.SIFT(nfeatures=0,nOctaveLayers=5,contrastThreshold=0.02,
                    edgeThreshold=10,sigma=0.5)
    k_tmp, des_tmp = sift.detectAndCompute(img,None)
#==============================================================================
# Plot the images and the keypoints
#==============================================================================
#    imagedot=cv2.drawKeypoints(img,k_tmp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#    cv2.imwrite('sift_keypoints.jpg',imagedot)
#    cv2.imwrite('original.jpg',img)
#==============================================================================
# Pack data together
#==============================================================================
    k_tmp, des_tmp = pack_keypoint(k_tmp,des_tmp)
    case_tag = np.reshape(np.asarray([tag for i in range(len(k_tmp))]),(len(k_tmp),1))    
    k_tmp = np.hstack((case_tag, k_tmp)) #add the case label to each feature list
    img_tag = np.asarray([number for i in range(len(k_tmp))])    
    k_tmp[:,-1] = img_tag #add img number to each feature
    if ii==1: 
        k = k_tmp
        des = des_tmp
    else:
        k = np.vstack((k,k_tmp))
        des = np.vstack((des,des_tmp))
#==============================================================================
#   Store keypoints  
# save data matrix [samples X features]
# features: case#,x,y,size,angle,response,octave,img_id, descriptor(128)
#==============================================================================
#write_features_to_file('./dataout/dataPoints_a594',k,des)
write_features_to_file('./dataout/dataPoints_cy5',k,des)

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#   Timing
#==============================================================================
#t= time.time()
#elapsed = time.time() - t
#print elapsed
#==============================================================================
# 
#==============================================================================
#    cv2.imwrite('rescaled_image.jpg',image)
#    img = cv2.drawKeypoints(image,kp)
#    cv2.imwrite(number+'_'+str(len(kp))+'.jpg',img)
#==============================================================================
# 
#==============================================================================
#    print os.path.join(dir_name, case, filename_prefix1 + "_" + number + 
#    "." + filename_suffix)
