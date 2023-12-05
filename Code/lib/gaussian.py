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


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x=None, eps=0.00001, time_last=True, meanval=None, minval=None, maxval=None, stdval=None):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        if meanval == None:
            self.userange = True
            x_np = x.detach().numpy()
            self.mean = torch.mean(x, 0)
            self.min = np.amin(x_np)
            self.max = np.amax(x_np)
            # self.std = torch.std(x, 0)
            self.std = torch.from_numpy(np.std(x_np, axis=0))
        else:
            self.userange = False
            self.mean = torch.tensor(meanval)
            self.min = torch.tensor(minval)
            self.max = torch.tensor(maxval)
            self.std = torch.tensor(stdval)

        self.eps = eps
        self.time_last = time_last # if the time dimension is the last dim

    def encode(self, x):
        if self.userange:
            return (x - self.min) / (self.max - self.min)
        else:
            return (x - self.mean) / self.std

    def decode(self, x):
        if self.userange:
            return (x * (self.max - self.min)) + self.min
        else:
            return x * self.std + self.mean

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


def gaussian_function(XX, ZZ, src):
    x_src, z_src = src[0], src[1]
    sigma2 = 0.0005
    grid = np.sqrt(1/sigma2 * ((XX-x_src)**2 + (ZZ-z_src)**2))
    grid = 1 / (sigma2 * np.sqrt(2*math.pi)) * (np.exp(-0.5 * grid**2))
    return grid


def gaussian_function_vectorized(XX, ZZ, src):
    sigma2 = 0.0005
    XX_sub = np.tile(XX, (len(src), 1, 1)) - src[:,1][:,np.newaxis,np.newaxis]
    ZZ_sub = np.tile(ZZ, (len(src), 1, 1)) - src[:,0][:,np.newaxis,np.newaxis]
    grid = np.sqrt(1/sigma2 * (XX_sub**2 + ZZ_sub**2))
    grid = 1 / (sigma2 * np.sqrt(2*np.pi)) * (np.exp(-0.5 * grid**2))
    return grid
