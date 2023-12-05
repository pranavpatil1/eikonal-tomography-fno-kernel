import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2
from lib.gaussian import *

class EikonalDatasetRaw(Dataset):
    def __init__(self, fid, dataset_type):
        train_test_split = int(len(fid["vel"]) * 5 / 6)

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

        horizontal_flip = v2.RandomHorizontalFlip(p=1)
        vertical_flip = v2.RandomVerticalFlip(p=1)
        rotate90 = v2.RandomRotation(degrees=(90, 90))
        rotate180 = v2.RandomRotation(degrees=(180, 180))
        rotate270 = v2.RandomRotation(degrees=(270, 270))

        self.flips = [horizontal_flip, vertical_flip]
        self.rotations = [rotate90, rotate180, rotate270]

    def __getitem__(self, i):
        # only apply transformations for odd index values
        transform_iter = True if i % 2 == 1 else False
        # maps incoming index to true array index
        i = i//2

        x_np = self.vels[i]
        y_np = self.kernels[i]
        source_np = gaussian_function(self.XX, self.ZZ, self.source_loc[i][::-1]/self.max_val)
        rec_np = gaussian_function(self.XX, self.ZZ, self.rec_loc[i][::-1]/self.max_val)

        if transform_iter:
            random_flip = np.random.choice([0, 1])
            x_np = np.flip(x_np, axis=random_flip)
            y_np = np.flip(y_np, axis=random_flip)
            source_np = np.flip(source_np, axis=random_flip)
            rec_np = np.flip(rec_np, axis=random_flip)

            random_rotation = np.random.choice([1, 2, 3])
            x_np = np.rot90(x_np, k=random_rotation)
            y_np = np.rot90(y_np, k=random_rotation)
            source_np = np.rot90(source_np, k=random_rotation)
            rec_np = np.rot90(rec_np, k=random_rotation)

        x_np = x_np.reshape(100, 100, 1)
        source_np = source_np.reshape(100, 100, 1)
        rec_np = rec_np.reshape(100, 100, 1)

        x_full_np = np.concatenate((x_np, source_np, rec_np), axis=2)

        x = torch.from_numpy(x_full_np.copy())
        y = torch.from_numpy(y_np.copy())

        # if transform_iter:
        #     # randomly choose a horizontal or vertical flip
        #     flip = np.random.choice(self.flips)
        #     # randomly choose a 90, 180, or 270 degree rotation
        #     rotate = np.random.choice(self.rotations)
        #     transform = v2.Compose([flip])

        #     x = transform(x)
        #     y = transform(torch.unsqueeze(y, 2))
        #     y = torch.squeeze(y, 2)

        return x, y

    def __len__(self):
        return len(self.vels) * 2

