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

from timeit import default_timer

from lib.utilities3 import *
from lib.datahelper import *
from lib.dataset_raw import *
from lib.fno import *

from lib.memorytracking import save_memory_usage
import datetime

torch.manual_seed(0)
np.random.seed(0)

# if we should utilize GPU cores
GPU = True
# if we should use smaller version of dataset
DATASET_BIG = True
# change this on every experiment
EXPERIMENT = "encoded"

if __name__ == '__main__':
    filename = '../Data/DatVel5000_Sou10_Rec10_Dim100x100_Gradient_Encoded.npz'
    if not DATASET_BIG:
        filename = '../Data/DatVel30_Sou100_Rec100_Dim100x100_Downsampled_Encoded.npz'
    
    memory_savefile = f"memprofiles/out-{EXPERIMENT}-{'gpu' if GPU else 'cpu'}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
    memory_tracker = []

    # shorthand
    savemem = lambda event: save_memory_usage(memory_tracker, event, gpu=GPU, savefile=memory_savefile)
    savemem("start")

    print ("loading dataset")
    with np.load(filename, allow_pickle=True) as fid:
        train_data = EikonalDatasetRaw(fid, 'train')
        test_data = EikonalDatasetRaw(fid, 'test')

    print ("finished loading dataset")
    
    savemem("load_dataset")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

    savemem("dataloader")

    sub = 1
    S = 100
    T_in = 10
    T = 40 # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;
    step = 1

    batch_size = 16
    learning_rate = 0.001
    epochs = 10
    iterations = epochs*(32//batch_size)

    modes = 6
    width = 128
    
    kernel_metadata_file='./lib/largefile-kernels-metadata.pkl'

    with open(kernel_metadata_file, 'rb') as f:
        kernel_metadata = pickle.loads(f.read())

    print ("finished loading metadata")

    y_normalizer = UnitGaussianNormalizer(
        meanval=kernel_metadata['mean'],
        stdval=kernel_metadata['std'],
        maxval=kernel_metadata['max'],
        minval=kernel_metadata['min'],
    )

    ######## TRAINING

    print ("begin training!!")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)
    test_loader_single = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    fno_model = FNO2d(modes, modes, width)
    if GPU:
        fno_model = fno_model.cuda()

    test_fno = []
    train_fno = []
    log_test_fno = []
    log_train_fno = []

    optimizer = torch.optim.AdamW(fno_model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    myloss = LpLoss(size_average=False)

    if GPU:
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.empty_cache()
        before_memory = torch.cuda.memory_allocated()

        t1 = default_timer()
        y_normalizer.cuda()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    savemem("begin_train")

    prefix = "modelSmall"
    if DATASET_BIG:
        prefix = "modelBig"

    for ep in range(1, epochs + 1):
        savemem(f"epoch{ep}")
        if ep % 50 == 0:
            torch.save(fno_model, f'/central/groups/mlprojects/eikonal/Models/{prefix}-mode' + str(modes) + '-width-' + str(width) + '-epoch-' + str(ep))

        fno_model.eval()
        test_l2 = 0.0

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

                curr_loss = myloss(out, y.view(batch_size,-1)).item()
                test_l2 += curr_loss

        fno_model.train()
        train_l2 = 0
        for x, y in train_loader:
            if GPU:
                x, y = x.cuda(), y.cuda()
            x, y = x.float(), y.float()

            optimizer.zero_grad()
            out = fno_model(x)
            if GPU:
                out = out.cuda()
            out = out.squeeze(-1)
            out = y_normalizer.decode(out)
            if GPU:
                out = out.cuda()
            y = y_normalizer.decode(y)
            if GPU:
                y = y.cuda()

            loss = myloss(out, y.view(batch_size,-1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_l2 += loss.item()

        train_l2/= 2500.0
        test_l2 /= 500.0
        print(f"epoch: {ep}, \tl2 train: {train_l2} \tl2 test: {test_l2}")


        train_fno.append(train_l2)
        test_fno.append(test_l2)
        log_train_fno.append(math.log(train_l2))
        log_test_fno.append(math.log(test_l2))
        save_losses(train_fno, test_fno, log_train_fno, log_test_fno, modes, width, ep)
        if ep % 50 == 0:
            test_losses = []
            with torch.no_grad():
                for x, y in test_loader_single:
                    if GPU:
                        x, y = x.cuda(), y.cuda()
                    x, y = x.float(), y.float()

                    out = fno_model(x)
                    out = out.squeeze(-1)
                    y = y_normalizer.decode(y)
                    if GPU:
                        y = y.cuda()

                    out = y_normalizer.decode(out)

                    curr_loss = myloss(out, y.view(1,-1)).item()
                    test_l2 += curr_loss
                    test_losses.append(curr_loss)

            plt.close('all')
            plot_loss_curves(train_fno, test_fno, log_train_fno, log_test_fno, modes, width, ep)
            # plot_residuals(test_losses, modes, width, ep)
            for i in range(len(train_fno)):
                print(f"epoch: {i}, \tl2 train: {train_fno[i]} \tl2 test: {test_fno[i]}")
            plt.show()


    t2 = default_timer()
    max_memory = torch.cuda.max_memory_allocated() - before_memory
    print(f'Max Memory Allocated: {max_memory} Time: {t2-t1}')
