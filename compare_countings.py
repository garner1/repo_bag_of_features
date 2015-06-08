import numpy as np
#import cv2
#import os
#from module import Massage
#import scipy
#import scipy.io
#import os
#==============================================================================
# Load data matrix [samples X features]
# features: case#,x,y,size,angle,response,octave,img_id, descriptor(128)
#==============================================================================

data_a594 = np.load('./dataout/filtered_a594.npy')

cases = np.unique(data_a594[:,0])
ii = 0
for case in cases:
    ii += 1
    rowboolean = data_a594[:,0]==case
    casedata = np.compress(rowboolean, data_a594, axis=0)
    images = np.unique(casedata[:,7])
    for image in images:
            rowboolean = casedata[:,7]==image
            imagedata = np.compress(rowboolean, casedata, axis=0)
            if ii==1: 
                output_a594 = [int(case), int(image), "1", np.shape(imagedata)[0]]
            else:
                output_a594 = np.vstack((output_a594,[int(case), int(image), "1", np.shape(imagedata)[0]]))
            print int(case), int(image), "1", np.shape(imagedata)[0]
        
    

data_cy5 = np.load('./dataout/filtered_cy5.npy')

cases = np.unique(data_cy5[:,0])
ii = 0
for case in cases:
    ii += 1
    rowboolean = data_cy5[:,0]==case
    casedata = np.compress(rowboolean, data_cy5, axis=0)
    images = np.unique(casedata[:,7])
    for image in images:
            rowboolean = casedata[:,7]==image
            imagedata = np.compress(rowboolean, casedata, axis=0)
            if ii==1: 
                output_cy5 = [int(case), int(image), "2", np.shape(imagedata)[0]]
            else:
                output_cy5 = np.vstack((output_cy5,[int(case), int(image), "2", np.shape(imagedata)[0]]))
            print int(case), int(image), "2", np.shape(imagedata)[0]

output = np.vstack((output_a594,output_cy5))
output = output.astype(np.float)
np.savetxt('mytable', output, fmt=['%u','%0.3u','%u','%u'] , delimiter=' ')