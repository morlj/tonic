from tonic.datasets import VPR
import numpy as np
import matplotlib.pyplot as plt

#create a transform function for the dataset
#Is 'max temporal distance'/filter time suitable?
# =============================================================================
# transform = transforms.Compose([#transforms.Denoise(filter_time=1000), #time=10000
#                                 #pull out hot pixels? using dataset txt file
#                                 transforms.DropEvent(0.9),
#                                 transforms.ToAveragedTimesurface(ordering=ordering,sensor_size=sensor_size), #for HATS
# #                               transforms.ToTimesurface(), #for HOTS
#                                 ])
# =============================================================================

#create the dataset variable
mydataset = VPR(save_to="./visual_place_recognition")

events_hats, events = mydataset[0]  #, imu, images 

timesurface_size = events_hats[0].shape[1]//2
imgSize = tuple(sum(x) for x in zip(mydataset.sensor_size[0:2],(timesurface_size*2,timesurface_size*2)))
imgVisualise = np.zeros(imgSize)
for index,event in enumerate(events_hats):
    x,y = events[index][1].astype(int), events[index][2].astype(int)
    min_x = x
    max_x = x + timesurface_size*2
    min_y = y
    max_y = y + timesurface_size*2
    imgVisualise[min_y:max_y+1,min_x:max_x+1] = imgVisualise[min_y:max_y+1,min_x:max_x+1] + event[0]
    
visNorm = np.linalg.norm(imgVisualise)
imgVisualiseGrey = imgVisualise / visNorm
plt.imshow(imgVisualiseGrey,cmap="gray")
plt.show()


#HATS - linear SVM on features from HOTS
#looking at LMTS?

#include test/train functions in VPR or leave here?


#Note: look at tonic.readthedocs.io/en/latest/transformations.html
