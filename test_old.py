# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:12:47 2015

@author: silvano
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:18:54 2015

@author: silvano
"""
import os
from module import Massage, pickle_keypoints, unpickle_keypoints, process_image
import subprocess 
import itertools
import matplotlib.pyplot as plt
import numpy as np
import time
import numpy as np
import cv2
import skimage
import cPickle as pickle
#import pandas as pd

t= time.time()
#==============================================================================
# Parameters
#==============================================================================
numb_of_tiff_images = 1
#==============================================================================
# Check if data file already exists
#==============================================================================
temp_array = []
if os.path.exists('./dataout/dataPoints.csv'): 
    answer = raw_input("Do you want to cancel the old data file?[y or n] ")
    if answer=='y': 
        subprocess.call(["mv","./dataout/dataPoints.csv","./dataout/dataPoints.csv.old"])
#==============================================================================
# Location of the tif files
#==============================================================================
dir_name = '/home/silvano/Work/BreastCancerAnalysis/data_input/Immagini_Casi_smFISH_Laura'
#==============================================================================
# Open the preselected list of valid files, to avoid bad tif files
#==============================================================================
f = open('./datain/mat_files_list.txt','r')
#==============================================================================
# for each line in the list of valid files
#==============================================================================
for line in itertools.islice(f, numb_of_tiff_images):
#==============================================================================
#     locate the path to the tif file
#==============================================================================
#    filename = f.readline()
    filename = line
    case = filename.split('/')[1]
    number = filename.split('/')[2].split('_')[1].split('.')[0]    
    filename_prefix1 = "a594"    
    filename_prefix2 = "cy5"
    filename_suffix = "tif"
    print os.path.join(dir_name, case, filename_prefix1 + "_" + number + 
    "." + filename_suffix)
    a594 = os.path.join(dir_name, case, filename_prefix1 + "_" + number + "." 
                    + filename_suffix)
    cy5 = os.path.join(dir_name, case, filename_prefix2 + "_" + number + "." 
                    + filename_suffix)
    imagefile = a594
#==============================================================================
#   Preprocess the file: 
#   image: 8bit image
#   original: 16bit projected along z image
#==============================================================================
    process_image(imagefile, 'output')
#    image, original = Massage(imagefile)
#==============================================================================
#   SIFT descriptor: good enough at point detection
#   these are good parameters: nfeatures=0,nOctaveLayers=5,
#                    contrastThreshold=0.02,
#                    edgeThreshold=10,sigma=0.5
#   kp[ind].pt or .size, .angle, .response, .octave, .class_id
#==============================================================================
#    sift = cv2.SIFT(nfeatures=0,nOctaveLayers=5,contrastThreshold=0.02,
#                    edgeThreshold=10,sigma=0.5)
#    kp, des = sift.detectAndCompute(image,None)
#    print 'file: ', number, '--SIFT key points number: ', len(kp)
#==============================================================================
#   Store keypoints    
#==============================================================================
#    temp = pickle_keypoints(kp, des)
#    temp_array.append(temp)
#==============================================================================
#   Create database
#==============================================================================
#pickle.dump(temp_array, open("keypoints_database.p", "wb"))
#==============================================================================
#   Retrieve Keypoint Features
#==============================================================================
#keypoints_database = pickle.load( open( "keypoints_database.p", "rb" ) )
#kp, desc = unpickle_keypoints(keypoints_database[0])

elapsed = time.time() - t
print elapsed
#==============================================================================
#   Timing
#==============================================================================
#t= time.time()
#elapsed = time.time() - t
#print elapsed
#==============================================================================
#    SURF descriptor: too bad at detecting points
#==============================================================================
#    surf = cv2.SURF(hessianThreshold=50, nOctaves=1, nOctaveLayers=1)
#    surf.upright = True
#    kp, des = surf.detectAndCompute(image,None)
#    img2 = cv2.drawKeypoints(image,kp,None,(255,0,0),4)
#    cv2.imwrite(number+'.jpg',img2)
#    print 'SURF key points number: ', len(kp)
#==============================================================================
# 
#==============================================================================
#    cv2.imwrite('rescaled_image.jpg',image)
#    img = cv2.drawKeypoints(image,kp)
#    cv2.imwrite(number+'_'+str(len(kp))+'.jpg',img)
