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
import pickle
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from IPython import display
import argparse

from timeit import default_timer

from lib.utilities3 import *
from lib.datahelper import *
from lib.dataset_raw import *
from lib.fno import *

from lib.memorytracking import save_memory_usage
import datetime

torch.manual_seed(0)
np.random.seed(0)

if __name__ == '__main__':
    MODES = 8
    WIDTH = 128
    BATCH_SIZE = 8

    # if we should use cuda
    GPU = True
    # if we should use full version of dataset
    DATASET_BIG = False
    EPOCHS = 1000
    # how often to save model/residual graphs
    SAVE_INTERVAL = 10

    # change this on every experiment
    EXPERIMENT = f"encoded-batchsize{BATCH_SIZE}-{'gpu' if GPU else 'cpu'}-{'BIG' if DATASET_BIG else 'SMALL'}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    print ("=" * 40)
    print ("RUN DETAILS")
    print ("=" * 40)

    print (f"MODES={MODES}, WIDTH={WIDTH}, BATCH SIZE={BATCH_SIZE}, GPU={GPU}, BIG DATASET={DATASET_BIG}, EPOCHS={EPOCHS}, SAVE INTERVAL={SAVE_INTERVAL}")
    print()

    print ("=" * 40)
    print ("BEGIN RUN")
    print ("=" * 40)

    filename = '../Data/DatVel5000_Sou10_Rec10_Dim100x100_Gradient_Encoded.npz'
    if not DATASET_BIG:
        filename = '../Data/DatVel30_Sou100_Rec100_Dim100x100_Downsampled_Encoded.npz'

    print ("loading dataset")
    with np.load(filename, allow_pickle=True) as fid:
        train_data = EikonalDatasetRaw(fid, 'train')
        test_data = EikonalDatasetRaw(fid, 'test')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_single = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    learning_rate = 0.0005

    kernel_metadata_file='./lib/largefile-kernels-metadata.pkl'

    with open(kernel_metadata_file, 'rb') as f:
        kernel_metadata = pickle.loads(f.read())

    y_normalizer = UnitGaussianNormalizer(
        meanval=kernel_metadata['mean'],
        stdval=kernel_metadata['std'],
        maxval=kernel_metadata['max'],
        minval=kernel_metadata['min'],
    )


    # Validation

    print ("begin validation!!")

    GPU = False
    fno_model = torch.load('../FinalizedModels/modelSmall-mode6-width-128-epoch-1000', map_location=torch.device('cpu'))    

    optimizer = torch.optim.AdamW(fno_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    myloss = LpLoss(size_average=False)

    if GPU:
        fno_model = fno_model.cuda()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.empty_cache()
        before_memory = torch.cuda.memory_allocated()
        y_normalizer.cuda()

    t1 = default_timer()
    test_l2 = 0
    with torch.no_grad():
        for x, y in test_loader:
            if GPU:
                x, y = x.cuda(), y.cuda()
            x, y = x.float(), y.float()
            out = fno_model(x)
            out = out.squeeze(-1)
            y = y_normalizer.decode(y)
            if GPU:
                y = y.cuda()

            out = y_normalizer.decode(out)

            curr_loss = myloss(out, y.view(len(y),-1)).item()
            test_l2 += curr_loss

        test_l2 /= len(test_data)

    t2 = default_timer()
    print(f"time: {t2-t1}, test l2 loss: {test_l2}, test log loss: {math.log(test_l2)}")

    if GPU:
        max_memory = torch.cuda.max_memory_allocated() - before_memory 
        print(f'Max Memory Allocated: {max_memory}')