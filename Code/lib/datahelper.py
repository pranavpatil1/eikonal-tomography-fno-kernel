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

from lib.gaussian import *

class EikonalDataset(Dataset):
    def __init__(self, fid, dataset_type):
        train_test_split = 2500
        max_val = 100
        if dataset_type == 'train':
            self.vels=torch.tensor(fid["vels"][:train_test_split])
            self.kernels=torch.tensor(fid["kernels"][:train_test_split])
            self.source_loc=torch.tensor(fid["source_loc"][:train_test_split])
            self.rec_loc=torch.tensor(fid["rec_loc"][:train_test_split])
        if dataset_type == 'test':
            self.vels=torch.tensor(fid["vels"][train_test_split:])
            self.kernels=torch.tensor(fid["kernels"][train_test_split:])
            self.source_loc=torch.tensor(fid["source_loc"][train_test_split:])
            self.rec_loc=torch.tensor(fid["rec_loc"][train_test_split:])
        self.XX, self.ZZ = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

        self.source_loc_gaussian=torch.tensor([gaussian_function(self.XX, self.ZZ, np.asarray(s)[::-1]/max_val) for s in self.source_loc])
        self.rec_loc_gaussian=torch.tensor([gaussian_function(self.XX, self.ZZ, np.asarray(s)[::-1]/max_val) for s in self.rec_loc])


    def __getitem__(self, i):
        # x = torch.from_numpy(np.asarray([np.float32(self.vels[i]), np.float32(gaussian_function(self.XX, self.ZZ, self.source_loc[i])),np.float32(gaussian_function(self.XX, self.ZZ, self.rec_loc[i]))]))
        x = torch.from_numpy(np.asarray([np.float32(self.vels[i]), np.float32(self.source_loc_gaussian[i]),np.float32(self.rec_loc_gaussian[i])]))
        y = torch.from_numpy(np.asarray([np.float32(self.kernels[i])]))

        return {'x': x, 'y': y, 'source': self.source_loc[i], 'receiver': self.rec_loc[i]}

    def __len__(self):
        return len(self.vels)



def save_losses(train_fno, test_fno, log_train_fno, log_test_fno, mode, width, epoch):
    dct = {"Train": train_fno, "Test": test_fno, "Log Train": log_train_fno, "Log Test": log_test_fno}
    df = pd.DataFrame(dct)
    stem = '/central/groups/mlprojects/eikonal/Losses/'
    filename = stem + 'lossSmall-mode-' + str(mode) + '-width-' + str(width) + ".csv"
    df.to_csv(filename)

def plot_loss_curves(train_fno, test_fno, log_train_fno, log_test_fno, mode, width, epoch):
    display.clear_output(wait=True)
    # display.display(pl.gcf())
    plt.close('all')
    x = [i for i in range(len(log_train_fno))]
    plt.plot(x[1::1], log_train_fno[1::1])
    stem = '/central/groups/mlprojects/eikonal/Losses/'
    filename = stem + 'lossPlotSmall-mode-' + str(mode) + '-width-' + str(width) + '-epoch-' + str(epoch)  + ".jpg"
    title = 'Loss Curves (Mode: ' + str(mode) + ' Width: ' + str(width) + ' Epoch: ' + str(epoch) + ")"
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss (L2)')
    plt.plot(x[1::1], log_test_fno[1::1])
    plt.legend(['Train', 'Test'])
    plt.savefig(filename)
    plt.show()

def plot_residuals(losses, mode, width, epoch, isTest=True):
    stem = '/central/groups/mlprojects/eikonal/Residuals/'
    filename = stem + 'residualsSmall-mode-' + str(mode) + '-width-' + str(width) + '-epoch-' + str(epoch)  + ".jpg"
    title = 'Residuals Histogram (Mode: ' + str(mode) + ' Width: ' + str(width) + ' Epoch: ' + str(epoch) + ")"
    plt.title(title)
    plt.xlabel('Image Loss')
    plt.ylabel('Amount')
    plt.hist(losses, bins=30)
    plt.legend('Train' if not isTest else 'Test')
    plt.savefig(filename)
    plt.show()