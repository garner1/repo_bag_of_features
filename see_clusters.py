# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:02:09 2015

@author: silvano
"""

import numpy as np
import cv2
import os
from module import Massage
#==============================================================================
# Load data
#==============================================================================
data_a594 = np.load('dataout/filtered_a594.npy')
labels = np.load('dataout/labels_a594_notScaled.npy')        
#data_cy5 = np.load('dataout/filtered_cy5.npy')
#==============================================================================
# Store all cases label in var cases, does not matter to use data_a594 or data_cy5
#==============================================================================
cases = np.unique(data_a594[:,0])
#==============================================================================
# Location of the tif files
#==============================================================================
dir_name = '/home/silvano/Work/BreastCancerAnalysis/data_input/Immagini_Casi_smFISH_Laura'
#==============================================================================
# Create a dictionary for case labels
#==============================================================================
dictionary = {6392:'220113_caso_6392_11_N2',6927:'010312_caso_6927_11_N2',11664:'110713_caso_11664_11_N1',9064:'250213_caso_9064_10_3N1',
559:'040713_caso_559_13_N1',5376:'200312_caso_5376_11_3N2',165:'050713_caso_165_11_3N1',3954:'170212_caso_3954_11_N1',
9959:'050313_caso_9959_12_N1',6555:'070212_caso_6555_10_4N1',375:'080313_caso_375_13_N2',3014:'280213_caso_3014_10_A1',
5918:'090713_caso_5918_11_5N1',8533:'080312_caso_8533_10_3N1',5071:'210213_caso_5071_10_N1',3594:'160312_caso_3594_11_NB1',
11198:'110313_caso_11198_12_N1',2203:'140312_caso_2203_11_N1',11200:'080213_caso_11200_11_N3',1689:'010213_caso_1689_11_3N1',
1630:'160212_caso_1630_11_N1',705:'110112_caso_705_11_4N1',740:'050213_caso_740_12_3N1',5933:'070213_caso_5933_11_4N1',
6120:'230113_caso_6120_11_2N1',1312:'170112_caso_1312_11_3N1',2350:'210312_caso_2350_11_3N1',1261:'110313_caso_1261_12_N2',
2156:'010313_caso_2156_12_RIS_N1',1517:'180112_caso_1517_11_3N2',10391:'060313_caso_10391_12_3N4',6600:'060213_caso_6600_10_6N1',
3995:'250113_caso_3995_11_2N1',1867:'100713_caso_1867_12_3N1',5030:'270213_caso_5030_10_1N1',1126:'070713_caso_1126_13_4N1',
913:'090713_caso_913_13_N4',11716:'070213_caso_11716_10_2N1',9291:'050313_caso_9291_12_N2',8878:'220213_caso_8878_10_N1',
6474:'200213_caso_6474_10_1N1',5597:'050213_caso_5597_11_3N1',8845:'110713_caso_8845_11_N1',7569:'210213_caso_7569_10_5N5',
1901:'280213_caso_1901_12_3N2',3209:'240212_caso_3209_11_4N1',12114:'120313_caso_12114_12_CAP1',475:'270212_caso_475_11_3N1',
10872:'060313_caso_10872_12_5N1'}
#==============================================================================
# plot filtered features
#==============================================================================
ii = 0 #counter in the loop
for case in cases:
    case = cases[1]
#    ii += 1
#    print 'case number '+ii+' of 49...'
#==============================================================================
#   filter features wrt cluster label
#==============================================================================
    cluster = 1    
    rowboolean = labels==cluster
    data = data_a594[np.where(rowboolean==True)[0]]
    rowboolean_a594 = data[:,0]==case 
#    rowboolean_cy5 = data[:,0]==case
    case_data_a594 = np.compress(rowboolean_a594, data, axis=0) 
#    case_data_cy5 = np.compress(rowboolean_cy5, data_cy5, axis=0)
    images = np.unique(case_data_a594[:,7]) 
    for img in images:
#        img = images[0]
        rowboolean_a594 = case_data_a594[:,7]==img 
#        rowboolean_cy5 = case_data_cy5[:,7]==img
        img_case_data_a594 = np.compress(rowboolean_a594, case_data_a594, axis=0) 
#        img_case_data_cy5 = np.compress(rowboolean_cy5, case_data_cy5, axis=0)
        kpts_a594 = img_case_data_a594[:,1:8] 
#        kpts_cy5 = img_case_data_cy5[:,1:8]
#==============================================================================
# encode the keypoints        
#==============================================================================
        keypoints_a594 = [cv2.KeyPoint(x, y, _size, _angle, _response, int(_octave), int(_class_id))
            for x, y, _size, _angle, _response, _octave, _class_id in list(kpts_a594)]
#        keypoints_cy5 = [cv2.KeyPoint(x, y, _size, _angle, _response, int(_octave), int(_class_id))
#            for x, y, _size, _angle, _response, _octave, _class_id in list(kpts_cy5)]
#==============================================================================
# open image location        
#==============================================================================
        case = int(case)
        img_numb = str(int(img)).zfill(3)
        filename_prefix1 = "a594"    
        filename_prefix2 = "cy5"
        filename_suffix = "tif"
        a594 = os.path.join(dir_name, dictionary[case], filename_prefix1 + "_" + img_numb + "." 
                    + filename_suffix)
        cy5 = os.path.join(dir_name, dictionary[case], filename_prefix2 + "_" + img_numb + "." 
                    + filename_suffix)
#==============================================================================
# draw images
#==============================================================================
        image_a594, original = Massage(a594)
#        image_cy5, original = Massage(cy5)
        imagedot_a594 = cv2.drawKeypoints(image_a594,keypoints_a594,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#        imagedot_cy5 = cv2.drawKeypoints(image_cy5,keypoints_cy5,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('keypoints_a594_' +'cluster'+str(cluster)+'_'+ img_numb + '.jpg', imagedot_a594)
#        cv2.imwrite('original_a594_' + img_numb + '.jpg', image_a594)
#        cv2.imwrite('keypoints_cy5_' + img_numb + '.jpg', imagedot_cy5)
#        cv2.imwrite('original_cy5_' + img_numb + '.jpg', image_cy5)


