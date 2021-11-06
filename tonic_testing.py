from tonic.datasets import VPR
import numpy as np
import time
from scipy.spatial import distance

import matplotlib.pyplot as plt

#create the dataset variable
mydataset = VPR(save_to="./visual_place_recognition")
tic = time.perf_counter()
query_trans,query_vis = mydataset[1]  #, imu, images
#ref_trans, ref_vis = mydataset[0] # this one is not synchronised with the other two
ref_trans, ref_vis = mydataset[2]
toc = time.perf_counter()
print(f"Event Frame ran in {toc-tic:0.4f} seconds")


## Visualise Event Frames

#for i, ref in enumerate(ref_vis)
plt.imshow(query_vis[5])
plt.show()
plt.imshow(ref_vis[5])
plt.show()

## Create descriptors for each place and visualise matches
descriptors = np.zeros([len(query_trans),len(ref_trans)])
for i, place_q in enumerate(query_trans):
    for j, place_r in enumerate(ref_trans):
        descriptor = distance.cdist(place_q[1,:,:],place_r[1,:,:],metric='cityblock') 
        descriptors[i,j] = np.trace(descriptor)


min_descriptors = descriptors.argmin(axis=1)
fig, ax = plt.subplots()
ax.imshow(descriptors,cmap='hot')
ax.plot(min_descriptors,np.arange(0,descriptors.shape[0]),'*',color='blue')
ax.set_xlabel('Reference Places')
ax.set_ylabel('Query Places')
plt.show()


