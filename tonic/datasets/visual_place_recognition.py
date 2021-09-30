import os
import numpy as np
#from importRosbag.importRosbag import importRosbag
import pandas as pd
from tonic.dataset import Dataset
from tonic import transforms
from tonic.download_utils import check_integrity, download_url


class VPR(Dataset):
    """Event-Based Visual Place Recognition With Ensembles of Temporal Windows <https://zenodo.org/record/4302805>.
    Events have (txyp) ordering.
    ::
    
        @article{fischer2020event,
          title={Event-based visual place recognition with ensembles of temporal windows},
          author={Fischer, Tobias and Milford, Michael},
          journal={IEEE Robotics and Automation Letters},
          volume={5},
          number={4},
          pages={6924--6931},
          year={2020},
          publisher={IEEE}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        download (bool): Choose to download data or verify existing files. If True and a file with the same
                    name and correct hash is already in the directory, download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.

    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, imu, images).
    """

#    base_url = "https://zenodo.org/record/4302805/files/"
    recordings = [  # recording names and their md5 hash
                  ["bags_2021-08-19-08-25-42_denoised.feather"],
                  ["bags_2021-08-19-08-28-43_denoised.feather"],
                  ["bags_2021-08-19-09-45-28_denoised.feather"],
    ]

    sensor_size = (260,346)
    ordering = "txyp"

    def __init__(self, save_to, download=True, transform=None, target_transform=None):
        super(VPR, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
#        print(self.location_on_system)
        
    def __getitem__(self, index):
        file_path = os.path.join(self.location_on_system, self.recordings[index])#[0])
        
        #each row is one event, event in txyp order
        #topics = importRosbag(filePathOrName=file_path, log="ERROR")
        event_stream = pd.read_feather(path=file_path)
        event_stream['t'] = (event_stream['t'] * 10e5).astype(np.uint64)  
        
        im_width, im_height = int(event_stream['x'].max() + 1), int(event_stream['y'].max() + 1)
        self.sensor_size = (im_width,im_height)
        
        events = np.copy(event_stream.to_numpy(np.uint64))
        imu = None #topics["/dvs/imu"]
        images = None #topics["/dvs/image_raw"]
        #         images["frames"] = np.stack(images["frames"])


#        if self.transform is not None:
#            events = self.transform(
#                events, self.sensor_size, self.ordering, images=images
#            )
            
        # now create and apply the transform
        # note that we are dropping a lot of events for HATS
        # HOTS works without dropping events
        # Try and find out what's going on
        
        #need to check how this fits with new version of tonic
        transform = transforms.Compose([
            transforms.DropEvent(0.9),
            transforms.ToAveragedTimesurface(ordering=self.ordering, sensor_size=self.sensor_size)
        ])

        # here we extract 1 second chunks from the numpy array
        place_number = 2
        # first find the absolute times
        time_start = events[0, 0] + place_number * 10e5
        time_end = events[0, 0] + (place_number + 1) * 10e5
    
        # then find the corresponding indices
        start_idx = np.searchsorted(events[:, 0], time_start)
        end_idx = np.searchsorted(events[:, 0], time_end)

        # and finally slice the array
        events_subset = np.copy(events[start_idx:end_idx])

        # and apply the transform
        out = transform(events_subset,self.ordering,self.sensor_size)    
            
            
            
            
        return out #events, imu, images

    def __len__(self):
        return len(self.recordings)

    def download(self):
        for (recording, md5_hash) in self.recordings:
            download_url(
                self.base_url + recording,
                self.location_on_system,
                filename=recording,
                md5=md5_hash,
            )
