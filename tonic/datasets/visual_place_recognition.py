import os
import numpy as np
#from importRosbag.importRosbag import importRosbag
import pandas as pd
from tonic.dataset import Dataset
from tonic import transforms
import gc

from tonic.download_utils import check_integrity, download_url


class VPR(Dataset):
    """Event-Based Visual Place Recognition With Ensembles of Temporal Windows <https://zenodo.org/record/4302805>.
    Events have (txyp) ordering.
    
    .. note::  To be able to read this dataset and its GPS files, you will need the `pynmea2` package installed. 

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
        transform (callable, optional): A callable of transforms to apply to the data.
    """

#    base_url = "https://zenodo.org/record/4302805/files/"
    recordings = [  # recording names and their md5 hash
                  ["bags_2021-08-19-08-25-42_denoised.feather"],
                  ["bags_2021-08-19-08-28-43_denoised.feather"],
                  ["bags_2021-08-19-09-45-28_denoised.feather"],
    ]

    sensor_size = (260,346,2) #xyp
    dtype = np.dtype([('t', 'uint64'), ('x', 'uint64'), ('y', 'uint64'), ('p', 'uint64')])
    ordering = dtype.names

    def __init__(self, save_to, transform=None, target_transform=None):
        super(VPR, self).__init__(save_to, transform=transform, target_transform=target_transform)

#        if not self._check_exists():
#            self.download()
        
    def __getitem__(self, index):
        file_path = os.path.join(self.location_on_system, self.recordings[index][0])
        
        #each row is one event, event in txyp order
        #topics = importRosbag(filePathOrName=file_path, log="ERROR")
        event_stream = pd.read_feather(path=file_path)
        event_stream['t'] = (event_stream['t']).astype(np.uint64)
#        event_stream['t'] -= event_stream['t'][0]
#        event_stream['t] *= 1e6
        
        im_width, im_height = int(event_stream['x'].max() + 1), int(event_stream['y'].max() + 1)
        im_npol = int(event_stream['p'].nunique())
        self.sensor_size = (im_width,im_height,im_npol)
        
        events = np.copy(event_stream.to_numpy(np.uint64))
        del event_stream ####try to avoid having to do this####
        imu = None #topics["/dvs/imu"]
        images = None #topics["/dvs/image_raw"]
        #         images["frames"] = np.stack(images["frames"])

#        if self.transform is not None:
#            events = self.transform(events)  #,self.sensor_size, self.ordering, images=images)
#        if self.target_transform is not None:
#            targets = self.target_transform(targets)


        # now create and apply the transform
        # note that we are dropping a lot of events for HATS
        # HOTS works without dropping events
        # Try and find out what's going on
        
        transform = transforms.Compose([
            transforms.ToAveragedTimesurface(sensor_size=self.sensor_size)
#           transforms.ToTimesurface() 
#           transforms.ToFrame()
        ])

        # here we extract 1 second chunks from the numpy array
        place_number = 2
        # first find the absolute times
        time_start = events[0,0] + place_number * 10e5
        time_end = events[0,0] + (place_number + 1) * 10e5
        
        # then find the corresponding indices
        start_idx = np.searchsorted(events[:,0], time_start)
        end_idx = np.searchsorted(events[:,0], time_end)

        # and finally slice the array
        events_subset = np.copy(events[start_idx:end_idx])
        events_subset = np.lib.recfunctions.unstructured_to_structured(events_subset, self.dtype)

        # and apply the transform
        out = transform(events_subset)
            
        return out, events_subset #, imu, images


    def __len__(self):
        return len(self.recordings)

    def download(self):
        for recording in self.recordings:
            for filename, md5_hash in recording:
                download_url(
                    self.base_url + filename,
                    self.location_on_system,
                    filename=filename,
                    md5=md5_hash,
                )

    def _check_exists(self):
        # check if all filenames are correct
        files_present = list(
            [
                check_integrity(os.path.join(self.location_on_system, filename))
                for recording in self.recordings
                for filename, md5 in recording
            ]
        )
        return all(files_present)