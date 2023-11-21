import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from IPython import display
from timeit import default_timer
from natsort import natsorted
import pickle

from lib.gaussian import gaussian_function, UnitGaussianNormalizer


vel_metadata_file='/groups/mlprojects/eikonal/Code/lib/largefile-vels-metadata.pkl'
kernel_metadata_file='/groups/mlprojects/eikonal/Code/lib/largefile-kernels-metadata.pkl'


class EikonalDatasetIdv(Dataset):
    def __init__(self, root_dir, dataset_type, transform=None):

        #partion based on train/test
        self.image_names = os.listdir(root_dir)        
        
        self.root_dir = root_dir
        self.transform = transform 
        self.image_names = natsorted(self.image_names)

        self.dataset_type = dataset_type

        self.XX, self.ZZ = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

        idxs = range(0, len(self.image_names))
        self.train_idx = np.random.choice(len(self.image_names), int(len(self.image_names)*0.8))
        self.test_idx = np.setdiff1d(idxs, self.train_idx)

        #save index split offline
        np.save("train_idxs.npy", self.train_idx)
        np.save("test_idxs.npy", self.test_idx)


        with open(vel_metadata_file, 'rb') as f:
            vel_metadata = pickle.loads(f.read())

        with open(kernel_metadata_file, 'rb') as f:
            kernel_metadata = pickle.loads(f.read())

        self.x_encoder = UnitGaussianNormalizer(
            meanval=vel_metadata['mean'],
            stdval=vel_metadata['std'],
            maxval=vel_metadata['max'],
            minval=vel_metadata['min'],
        )

        self.y_encoder = UnitGaussianNormalizer(
            meanval=kernel_metadata['mean'],
            stdval=kernel_metadata['std'],
            maxval=kernel_metadata['max'],
            minval=kernel_metadata['min'],
        )


    #need to encorporate encoding here
    def __getitem__(self, i):
        img_path = os.path.join(self.root_dir, self.image_names[i])
        file = np.loadz(img_path)
        max_val = 100

        vels=torch.tensor(file["vels"])
        kernel=torch.tensor(file["kernels"])
        source_loc=torch.tensor(file["source_loc"])
        rec_loc=torch.tensor(file["rec_loc"])

        source_loc = torch.tensor(gaussian_function(self.XX, self.ZZ, np.asarray(source_loc)[::-1]/max_val))
        rec_loc = torch.tensor(gaussian_function(self.XX, self.ZZ, np.asarray(rec_loc)[::-1]/max_val))

        vels = self.x_encoder.encode(vels)
        kernel = self.y_encoder.encode(kernel)

        x = torch.from_numpy(np.asarray([np.float32(vels), np.float32(source_loc),np.float32(rec_loc)]))
        y = torch.from_numpy(np.asarray([np.float32(kernel)]))

        return {'x': x, 'y': y}

    def __len__(self):
        return len(self.image_names)
    





