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
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  12, 12, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width) # TODO: when s & r are added, change 1 to 3
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = self.q(x)

        x = x.permute(0, 2, 3, 1)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()


        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class EikonalDataset(Dataset):
    def __init__(self, root_dir, dataset_type, transform=None):

        #partion based on train/test
        self.image_names = os.listdir(root_dir)
        
        self.root_dir = root_dir
        self.transform = transform 
        self.image_names = natsorted(image_names)

        self.dataset_type = dataset_type

        self.XX, self.ZZ = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))


    def __getitem__(self, i):
        img_path = os.path.join(self.root_dir, self.image_names[i//2])
        file = np.loadz(img_path)
        max_val = 100

        if self.dataset_type == 'train':
            vels=torch.tensor(file["vels"])
            kernel=torch.tensor(file["kernels"])
            source_loc=torch.tensor(file["source_loc"])
            rec_loc=torch.tensor(file["rec_loc"])

            source_loc = torch.tensor(gaussian_function(self.XX, self.ZZ, np.asarray(source_loc)[::-1]/max_val))
            rec_loc = torch.tensor(gaussian_function(self.XX, self.ZZ, np.asarray(rec_loc)[::-1]/max_val))


        x = torch.from_numpy(np.asarray([np.float32(vels), np.float32(source_loc),np.float32(rec_loc)]))
        y = torch.from_numpy(np.asarray([np.float32(kernel)]))

        return {'x': x, 'y': y, 'source': self.source_loc[i], 'receiver': self.rec_loc[i]}

    def __len__(self):
        return len(self.image_names)*2 
    


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        x_np = x.detach().numpy()
        self.mean = torch.mean(x, 0)
        self.min = np.amin(x_np)
        self.max = np.amax(x_np)
        # self.std = torch.std(x, 0)
        self.std = torch.from_numpy(np.std(x_np, axis=0))

        self.eps = eps
        self.time_last = time_last # if the time dimension is the last dim

        print (torch.min(self.mean), torch.max(self.mean))

        print (torch.min(self.std), torch.max(self.std))

    def encode(self, x):
        x = (x - self.min) / (self.max - self.min)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                    std = self.std[...,sample_idx] + self.eps # T*batch*n
                    mean = self.mean[...,sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * (self.max - self.min)) + self.min
        return x

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


def save_losses(train_fno, test_fno, log_train_fno, log_test_fno, mode, width, epoch):
    dct = {"Train": train_fno, "Test": test_fno, "Log Train": log_train_fno, "Log Test": log_test_fno}
    df = pd.DataFrame(dct)
    stem = '/central/groups/mlprojects/eikonal/Losses/'
    filename = stem + 'lossBig-mode-' + str(mode) + '-width-' + str(width) + ".csv"
    df.to_csv(filename)

def plot_loss_curves(train_fno, test_fno, log_train_fno, log_test_fno, mode, width, epoch):
    display.clear_output(wait=True)
    # display.display(pl.gcf())
    plt.close('all')
    x = [i for i in range(len(log_train_fno))]
    plt.plot(x[1::1], log_train_fno[1::1])
    stem = '/central/groups/mlprojects/eikonal/Losses/'
    filename = stem + 'lossPlotBig-mode-' + str(mode) + '-width-' + str(width) + '-epoch-' + str(epoch)  + ".jpg"
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
    filename = stem + 'residualsBig-mode-' + str(mode) + '-width-' + str(width) + '-epoch-' + str(epoch)  + ".jpg"
    title = 'Residuals Histogram (Mode: ' + str(mode) + ' Width: ' + str(width) + ' Epoch: ' + str(epoch) + ")"
    plt.title(title)
    plt.xlabel('Image Loss')
    plt.ylabel('Amount')
    plt.hist(losses, bins=30)
    plt.legend('Train' if not isTest else 'Test')
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':

    outputfilename = 'train'
    output_file = open(outputfilename + '.txt', 'w')
    sys.stdout = output_file

    filename = '/central/scratch/ebiondi/DatVel6000_Sou10_Rec10_Dim100x100_Gaussian_scale4.npz'
    with np.load(filename, allow_pickle=True) as fid:
        train_data = EikonalDataset(fid[0:3000], 'train')
        print(train_data)
        test_data = EikonalDataset(fid[0:3000], 'test')

    # with np.load(filename, allow_pickle=True) as fid:
    # vels=fid["vels"][1000,:,:]
    # kernels=fid["kernels"][1000,:,:]
    # source_loc=fid["source_loc"][1000,:]
    # rec_loc=fid["rec_loc"][1000,:]



    x = np.linspace(0, 1, 100)
    z = np.linspace(0, 1, 100)

    XX, ZZ = np.meshgrid(x, z)
    x_coordinate = 25.5/100
    y_coordinate = 75.8/100

    gf = gaussian_function(XX, ZZ, (x_coordinate, y_coordinate))

    x_train_vels = train_data.vels

    x_train_src = train_data.source_loc_gaussian
    x_train_rec = train_data.rec_loc_gaussian
    y_train = train_data.kernels

    x_test_vels = test_data.vels
    x_test_src = test_data.source_loc_gaussian
    x_test_rec = test_data.rec_loc_gaussian
    y_test = test_data.kernels


    x_normalizer = UnitGaussianNormalizer(x_train_vels)
    x_train_vels = x_normalizer.encode(x_train_vels)


    x_test_vels = x_normalizer.encode(x_test_vels)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    y_test = y_normalizer.encode(y_test)


    x_train_vels = x_train_vels.reshape(len(x_train_vels),100,100,1)
    x_test_vels = x_test_vels.reshape(len(x_test_vels),100,100,1)

    x_train_src = x_train_src.reshape(len(x_train_src),100,100,1) 
    x_test_src = x_test_src.reshape(len(x_test_src),100,100,1)

    x_train_rec = x_train_rec.reshape(len(x_train_rec),100,100,1) 
    x_test_rec = x_test_rec.reshape(len(x_test_rec),100,100,1)

    x_train = torch.cat((x_train_vels, x_train_src, x_train_rec), 3)
    x_test = torch.cat((x_test_vels, x_test_src, x_test_rec), 3)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=16, shuffle=False)


    ######## TRAINING

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
    width = 64

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    test_loader_single = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)

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
    y_normalizer.cuda()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for ep in range(1, epochs + 1):
        if ep % 20 == 0:
            torch.save(fno_model, '/central/groups/mlprojects/eikonal/Models/modelBig-mode' + str(modes) + '-width-' + str(width) + '-epoch-' + str(ep))

        fno_model.eval()
        test_l2 = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                x, y = x.float(), y.float()

                out = fno_model(x)
                out = out.squeeze(-1)
                y = y_normalizer.decode(y).cuda()

                out = y_normalizer.decode(out)

                curr_loss = myloss(out, y.view(batch_size,-1)).item()
                test_l2 += curr_loss

        fno_model.train()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            x, y = x.float(), y.float()

            optimizer.zero_grad()
            out = fno_model(x).cuda()
            out = out.squeeze(-1)
            out = y_normalizer.decode(out).cuda()
            y = y_normalizer.decode(y).cuda()

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
                    x, y = x.cuda(), y.cuda()
                    x, y = x.float(), y.float()

                    out = fno_model(x)
                    out = out.squeeze(-1)
                    y = y_normalizer.decode(y).cuda()

                    out = y_normalizer.decode(out)

                    curr_loss = myloss(out, y.view(1,-1)).item()
                    test_l2 += curr_loss
                    test_losses.append(curr_loss)

            plt.close('all')
            plot_loss_curves(train_fno, test_fno, log_train_fno, log_test_fno, modes, width, ep)
            plot_residuals(test_losses, modes, width, ep)
            for i in range(len(train_fno)):
                print(f"epoch: {i}, \tl2 train: {train_fno[i]} \tl2 test: {test_fno[i]}")
            plt.show()


    t2 = default_timer()
    max_memory = torch.cuda.max_memory_allocated() - before_memory
    print(f'Max Memory Allocated: {max_memory} Time: {t2-t1}')