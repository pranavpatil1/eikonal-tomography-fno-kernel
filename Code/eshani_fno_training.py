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

from lib.utilities3 import *
from lib.datahelper import *
from lib.fno import *
from lib.dataset_indiv import EikonalDatasetIdv
import datetime

# from lib.memorytracking import save_memory_usage
# import datetime

torch.manual_seed(0)
np.random.seed(0)

DATASET_BIG = True
GPU = True
EXPERIMENT = "encoded_eshani_indiv_big"
EXPERIMENT = EXPERIMENT + f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
SAVE_INTERVAL = 5

if __name__ == '__main__':

    filename = '/central/scratch/ebiondi/DatVel6000_Sou10_Rec10_Dim100x100_Gaussian_scale4/' 
    if not DATASET_BIG:
        filename = '../Data/DatVel30_Sou100_Rec100_Dim100x100_Downsampled.npz'
    
    print ("loading dataset")
    train_data = EikonalDatasetIdv(filename, 'train', setSeed=10)
    test_data = EikonalDatasetIdv(filename, 'test', setSeed=10)

    print ("finished loading dataset")


    ######## TRAINING

    print ("begin training!!")

    sub = 1
    S = 100
    T_in = 10
    T = 40 # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;
    step = 1

    batch_size = 64
    learning_rate = 0.001
    epochs = 1000
    iterations = epochs*(32//batch_size)

    modes = 6
    width = 64

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)
    test_loader_single = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)


    fno_model = FNO2d(modes, modes, width).cuda()

    test_fno = []
    train_fno = []
    log_test_fno = []
    log_train_fno = []

    optimizer = torch.optim.AdamW(fno_model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    myloss = LpLoss(size_average=False)

    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()
    before_memory = torch.cuda.memory_allocated()

    t1 = default_timer()
    train_data.y_encoder.cuda()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    prefix = "model-small-eshani"
    if DATASET_BIG:
        prefix = "model-big-eshani"
    
    prefix += f"-{EXPERIMENT}"
    print("begin training loop")
    for ep in range(1, epochs + 1):
        if ep % SAVE_INTERVAL == 0:
            save_models(fno_model, ep, modes, width, prefix=prefix)
            
            
        fno_model.eval()
        test_l2 = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                x, y = x.float(), y.float()

                out = fno_model(x)
                out = out.squeeze(-1)
                y = test_data.y_encoder.decode(y).cuda()

                out = test_data.y_encoder.decode(out)

                curr_loss = myloss(out, y.view(len(y),-1)).item()
                test_l2 += curr_loss

        fno_model.train()
        train_l2 = 0
        
        print("calculating losses")
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            x, y = x.float(), y.float()

            optimizer.zero_grad()
            out = fno_model(x).cuda()
            out = out.squeeze(-1)
            out = train_data.y_encoder.decode(out).cuda()
            y = train_data.y_encoder.decode(y).cuda()

            loss = myloss(out, y.view(len(y),-1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_l2 += loss.item()

        train_l2/= 2500.0
        test_l2 /= 500.0
        print(f"eshani, epoch: {ep}, \tl2 train: {train_l2} \tl2 test: {test_l2}")


        train_fno.append(train_l2)
        test_fno.append(test_l2)
        log_train_fno.append(math.log(train_l2))
        log_test_fno.append(math.log(test_l2))
        save_losses(train_fno, test_fno, log_train_fno, log_test_fno, modes, width, prefix=prefix)
        if ep % SAVE_INTERVAL == 0:
            test_losses = []
            with torch.no_grad():
                for x, y in test_loader_single:
                    x, y = x.cuda(), y.cuda()
                    x, y = x.float(), y.float()

                    out = fno_model(x)
                    out = out.squeeze(-1)
                    y = test_data.y_encoder.decode(y).cuda()

                    out = test_data.y_encoder.decode(out)

                    curr_loss = myloss(out, y.view(1,-1)).item()
                    test_l2 += curr_loss
                    test_losses.append(curr_loss)

            plt.close('all')
            plot_loss_curves(train_fno, test_fno, log_train_fno, log_test_fno, modes, width, ep, prefix="bigESHANI")
            for i in range(len(train_fno)):
                print(f"eshani, epoch: {i}, \tl2 train: {train_fno[i]} \tl2 test: {test_fno[i]}")
            plt.show()


    t2 = default_timer()
    max_memory = torch.cuda.max_memory_allocated() - before_memory
    print(f'Max Memory Allocated: {max_memory} Time: {t2-t1}')