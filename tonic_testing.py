from tonic.datasets import VPR
import tonic.transforms as transforms

#probably can delete this now -- added to dataset function
#dvs_data = pd.read_feather('./visual_place_recognition/dvs_vpr_2020-04-21-17-03-03_subset.feather')
#dvs_subset = dvs_data.to_numpy()
#ordering  = "txyp"

#create a transform function for the dataset
#Is 'max temporal distance'/filter time suitable?
transform = transforms.Compose([#transforms.Denoise(filter_time=100), #time=10000
                                #pull out hot pixels? using dataset txt file
                                transforms.ToAveragedTimesurface(cell_size=10,surface_size=3), #for HATS
#                               transforms.ToTimesurface(), #for HOTS
                                ])

#create the dataset variable
mydataset = VPR(save_to="./visual_place_recognition",download=False,transform=transform)

events, imu, images = mydataset[0]

#HATS - linear SVM on features from HOTS
#looking at LMTS?

#include test/train functions in VPR or leave here?


#Note: look at tonic.readthedocs.io/en/latest/transformations.html
