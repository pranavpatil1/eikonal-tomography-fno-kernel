import numpy as np
from torch.utils.data import Dataset
from lib.gaussian import *

class EikonalDatasetRaw(Dataset):
    def __init__(self, fid, dataset_type):
        train_test_split = 2500

        self.max_val = 100

        if dataset_type == 'train':
            self.vels = fid["vel"][:train_test_split]
            self.kernels = fid["kernels"][:train_test_split]
            self.source_loc = fid["source_loc"][:train_test_split]
            self.rec_loc = fid["rec_loc"][:train_test_split]
        elif dataset_type == 'test':
            self.vels = fid["vel"][train_test_split:]
            self.kernels = fid["kernels"][train_test_split:]
            self.source_loc = fid["source_loc"][train_test_split:]
            self.rec_loc = fid["rec_loc"][train_test_split:]

        self.XX, self.ZZ = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

    def __getitem__(self, i):
        x_np = self.vels[i]
        y_np = self.kernels[i]
        source_np = gaussian_function(self.XX, self.ZZ, self.source_loc[i][::-1])/self.max_val
        rec_np = gaussian_function(self.XX, self.ZZ, self.rec_loc[i][::-1])/self.max_val

        x_np = x_np.reshape(100, 100, 1)
        source_np = x_np.reshape(100, 100, 1)
        rec_np = x_np.reshape(100, 100, 1)

        x_full_np = np.concatenate((x_np, source_np, rec_np), axis=2)

        # x = torch.from_numpy()
        # y = torch.from_numpy()
        # source = 
        # rec = gaussian_function(self.XX, self.ZZ, self.rec_loc[i][::-1])/max_val

        # print (x_np.shape)
        # print (source_np.shape)
        # print (rec_np.shape)
        # print (x_full_np.shape)
        # print (y_np.shape)

        return x_full_np, y_np

    def __len__(self):
        return len(self.vels)

