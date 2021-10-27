from tonic.datasets import VPR
import tonic.transforms as transforms


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
transform = None

#create the dataset variable
mydataset = VPR(save_to="./visual_place_recognition",download=False,transform=transform)

events = mydataset[0]  #, imu, images
a=4
z=1

#HATS - linear SVM on features from HOTS
#looking at LMTS?

#include test/train functions in VPR or leave here?


#Note: look at tonic.readthedocs.io/en/latest/transformations.html
