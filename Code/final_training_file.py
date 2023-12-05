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
from lib.fno import *

from lib.memorytracking import save_memory_usage
import datetime

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(
                    prog='Eikonal FNO Training',
                    description='Trains an FNO model to learn kernels')

parser.add_argument('-m', '--modes', type=int)
parser.add_argument('-w', '--width', type=int)
parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('-e', '--epochs', type=int, default=250)
parser.add_argument('-s', '--saveinterval', type=int, default=10)
parser.add_argument('-c', '--checkpoint', type=str, default='')

# bool args
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--smalldataset', action="store_true")
parser.add_argument('--transforms', action="store_true")

# load everything
args = parser.parse_args()

MODES = args.modes
WIDTH = args.width
BATCH_SIZE = args.batch_size
CHECKPOINT_FILE = args.checkpoint

# if we should use cuda
GPU = False if args.cpu else True
# if we should use full version of dataset
DATASET_BIG = False if args.smalldataset else True 
# if we should data augment with transforms
TRANSFORMS = False if args.transforms else True
if TRANSFORMS:
    from lib.dataset_raw_transform import *
else:
    from lib.dataset_raw import *
# how many epochs to run for
EPOCHS = args.epochs
# how often to save model/residual graphs
SAVE_INTERVAL = args.saveinterval

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

memory_savefile = f"memprofiles/out-{EXPERIMENT}.pkl"
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

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader_single = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

savemem("dataloader")

learning_rate = 0.0005

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
if CHECKPOINT_FILE == '':
    fno_model = FNO2d(MODES, MODES, WIDTH)
else:
    fno_model = torch.load(CHECKPOINT_FILE)
if GPU:
    fno_model = fno_model.cuda()

test_fno = []
train_fno = []
log_test_fno = []
log_train_fno = []

optimizer = torch.optim.AdamW(fno_model.parameters(), lr=learning_rate, weight_decay=1e-4)
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

prefix = EXPERIMENT

for ep in range(1, EPOCHS + 1):
    savemem(f"epoch{ep}")
    if ep < 5 or ep % SAVE_INTERVAL == 0:
        save_models(fno_model, ep, MODES, WIDTH, prefix=prefix)

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

            curr_loss = myloss(out, y.view(len(y),-1)).item()
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

        loss = myloss(out, y.view(len(y),-1))
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_l2 += loss.item()

    train_l2/= len(train_data)
    test_l2 /= len(test_data)
    print(f"epoch: {ep}, \tl2 train: {train_l2} \tl2 test: {test_l2}")

    train_fno.append(train_l2)
    test_fno.append(test_l2)
    log_train_fno.append(math.log(train_l2))
    log_test_fno.append(math.log(test_l2))

    save_losses(train_fno, test_fno, log_train_fno, log_test_fno, MODES, WIDTH, prefix=prefix)

    if ep < 5 or ep % SAVE_INTERVAL == 0:
        plot_loss_curves(train_fno, test_fno, log_train_fno, log_test_fno, MODES, WIDTH, ep, prefix=prefix)
        for i in range(len(train_fno)):
            print(f"epoch: {i}, \tl2 train: {train_fno[i]} \tl2 test: {test_fno[i]}")

t2 = default_timer()
max_memory = torch.cuda.max_memory_allocated() - before_memory
print(f'Max Memory Allocated: {max_memory} Time: {t2-t1}')
