from tonic.datasets import VPR
import numpy as np
import matplotlib.pyplot as plt
import time


#moving this in to vpr so that whole dataset can be returned
def visualiseEvents(events, vprObj=None, datastream=None):
    if (vprObj is None) and (datastream is None):
        imgVisualiseGrey = events.T
    else:
        timesurface_size = events[0].shape[1]//2
        imgSize = tuple(sum(x) for x in zip(vprObj.sensor_size[0:2],(timesurface_size*2,timesurface_size*2)))
        imgVisualise = np.zeros(imgSize)
        for index,event in enumerate(events):
            x,y = datastream[index][1].astype(int), datastream[index][2].astype(int)
            min_x = x
            max_x = x + timesurface_size*2
            min_y = y
            max_y = y + timesurface_size*2
            imgVisualise[min_x:max_x+1,min_y:max_y+1] = imgVisualise[min_x:max_x+1,min_y:max_y+1] + event[0]
        visNorm = np.linalg.norm(imgVisualise)
        imgVisualiseGrey = imgVisualise / visNorm
        
    plt.imshow(imgVisualiseGrey.T,cmap="gray")
    plt.show()
    return imgVisualiseGrey


#create the dataset variable
mydataset = VPR(save_to="./visual_place_recognition")
tic = time.perf_counter()
query_trans,query_events = mydataset[0]  #, imu, images 
ref_trans, ref_events = mydataset[1]
toc = time.perf_counter()

## For Frame ##########################
visualiseEvents(ref_trans[0][0])
visualiseEvents(query_trans[0][0])
## For HATS and HOTS ##################
#visualiseEvents(ref_trans,mydataset,ref_events)
#visualiseEvents(query_trans,mydataset,query_events)
print(f"HOTS ran in {toc-tic:0.4f} seconds")


#HATS - linear SVM on 'image' array