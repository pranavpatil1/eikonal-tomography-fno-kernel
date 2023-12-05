import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import pickle
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

stem = '/central/groups/mlprojects/eikonal/Visualizations'

def mkdir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def model_info(model_file):
    model_info = model_file.split("Experiments")
    model_info = os.path.split(model_info[1])
    model_expirement = model_info[0]
    model_params = model_info[1]
    return model_expirement, model_params

def plot(model_file, model, data, plot_type):

    model_expirement, model_params = model_info(model_file)
    model.eval()

    kernel_metadata_file='/central/groups/mlprojects/eikonal/Code/lib/largefile-kernels-metadata.pkl'

    with open(kernel_metadata_file, 'rb') as f:
        kernel_metadata = pickle.loads(f.read())

    y_normalizer = UnitGaussianNormalizer(
        meanval=kernel_metadata['mean'],
        stdval=kernel_metadata['std'],
        maxval=kernel_metadata['max'],
        minval=kernel_metadata['min'],
    )

    vel_metadata_file='/central/groups/mlprojects/eikonal/Code/lib/largefile-vels-metadata.pkl'

    with open(vel_metadata_file, 'rb') as f:
        vel_metadata = pickle.loads(f.read())

    x_normalizer = UnitGaussianNormalizer(
        meanval=vel_metadata['mean'],
        stdval=vel_metadata['std'],
        maxval=vel_metadata['max'],
        minval=vel_metadata['min'],
    )

    filedir = f'{stem}{model_expirement}/{model_params}/'
    filename = f'{stem}{model_expirement}/{model_params}/{plot_type}_visual.jpg'

    data_indices = [0, 100, 200, 300]
    fig, axs = plt.subplots(len(data_indices), 5, figsize=(15, len(data_indices) * 2))

    for i in range(len(data_indices)):
        x, y = data.__getitem__(data_indices[i])
        x, y = x.float().unsqueeze(0), y_normalizer.decode(y.float().unsqueeze(0))
        y_pred = y_normalizer.decode(model(x)[0].squeeze(-1)).detach().numpy()
        x, y = x.squeeze(0).detach().numpy(), y.squeeze(0).detach().numpy()
        
        vfd = x_normalizer.decode(torch.tensor(x[:, :, 0]))
        src = x[:, :, 1]
        rec = x[:, :, 2]
        truth = y
        pred = y_pred

        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0: 
                return v
            return v / norm
        difference = abs(truth - pred)

        min_kernel = min(np.min(pred), np.min(truth))
        max_kernel = max(np.max(pred), np.max(truth))

        axs[i, 0].imshow(vfd)
        plt.colorbar(axs[i, 0].imshow(vfd))
        axs[i, 1].imshow(src - rec)
        axs[i, 2].imshow(truth, vmin=min_kernel, vmax=max_kernel)
        plt.colorbar(axs[i, 2].imshow(truth, vmin=min_kernel, vmax=max_kernel))
        axs[i, 3].imshow(pred, vmin=min_kernel, vmax=max_kernel)
        plt.colorbar(axs[i, 3].imshow(pred, vmin=min_kernel, vmax=max_kernel))
        axs[i, 4].imshow(difference, cmap='PiYG')
        plt.colorbar(axs[i, 4].imshow(difference, cmap='PiYG'))


        for j in range(4):
            axs[i, j].axis('off')
            axs[i, j].grid(False)

        if i == 0:
            axs[i, 0].set_title('velocity fields')
            axs[i, 1].set_title('source receiver pair')
            axs[i, 2].set_title('true kernel')
            axs[i, 3].set_title('predicted kernel')
            axs[i, 4].set_title('difference plot')

    mkdir_if_not_exists(filedir)
    fig.suptitle(f'{model_params} ({plot_type})')
    plt.savefig(filename)
    print(f'{plot_type} saved to {filename}')
