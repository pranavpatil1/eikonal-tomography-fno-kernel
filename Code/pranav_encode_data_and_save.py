import pickle
import numpy as np
from lib.datahelper import UnitGaussianNormalizer
from lib.gaussian import gaussian_function_vectorized, gaussian_function

TEST = True

vel_metadata_file='./lib/largefile-vels-metadata.pkl'
kernel_metadata_file='./lib/largefile-kernels-metadata.pkl'

filename = '../Data/DatVel5000_Sou10_Rec10_Dim100x100_Gradient.npz'
outfile = '../Data/DatVel5000_Sou10_Rec10_Dim100x100_Gradient_Encoded.npz'

if TEST:
    filename = '../Data/DatVel30_Sou100_Rec100_Dim100x100_Downsampled.npz'
    outfile = '../Data/DatVel30_Sou100_Rec100_Dim100x100_Downsampled_Encoded.npz'

print ("starting!")

with open(vel_metadata_file, 'rb') as f:
    vel_metadata = pickle.loads(f.read())

with open(kernel_metadata_file, 'rb') as f:
    kernel_metadata = pickle.loads(f.read())

print ("finished loading metadata")

x_encoder = UnitGaussianNormalizer(
    meanval=vel_metadata['mean'],
    stdval=vel_metadata['std'],
    maxval=vel_metadata['max'],
    minval=vel_metadata['min'],
)

y_encoder = UnitGaussianNormalizer(
    meanval=kernel_metadata['mean'],
    stdval=kernel_metadata['std'],
    maxval=kernel_metadata['max'],
    minval=kernel_metadata['min'],
)

print ("created encoder")

with np.load(filename, mmap_mode='c') as fid:
    X_on_disk = fid["vels"]
    x_encoded = x_encoder.encode(X_on_disk)

    Y_on_disk = fid["kernels"]
    y_encoded = y_encoder.encode(Y_on_disk)

    print ("finished encoding")

    XX, ZZ = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    max_val = 100

    # source_loc_gaussian=gaussian_function_vectorized(XX, ZZ, fid["source_loc"]/max_val)
    # rec_loc_gaussian=gaussian_function_vectorized(XX, ZZ, fid["rec_loc"]/max_val)
    
    # print ("finished gaussian")

    # test = gaussian_function(XX, ZZ, fid["source_loc"][0,:][::-1]/max_val)
    # print (f"check validity! {np.array_equal(test, source_loc_gaussian[0])}")

    with open(outfile, 'wb') as outfile:
        np.savez(
            outfile,
            vel=x_encoded,
            kernels=y_encoded,
            source_loc=fid["source_loc"],
            rec_loc=fid["rec_loc"],
        )

    print ("saved output!")