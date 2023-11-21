import pickle
import numpy as np 

import gc
import os

import numpy as np
import psutil

process = psutil.Process(os.getpid())

# filename = '/central/groups/mlprojects/eikonal/Data/DatVel30_Sou100_Rec100_Dim100x100_Downsampled.npz'
filename = '/central/scratch/ebiondi/DatVel5000_Sou10_Rec10_Dim100x100_Gradient.npz'

with np.load(filename, mmap_mode='c') as fid:
    for target in ["vels", "kernels"]:
        print (f"working on {target}")
        
        outfile=f'largefile-{target}-metadata.pkl'

        X_on_disk = fid[target]
        results = {}
        print ("starting!")

        results['mean'] = np.nanmean(X_on_disk)
        with open(outfile, 'wb') as file: 
            pickle.dump(results, file)
        print ('got mean')

        results['std'] = np.nanstd(X_on_disk)
        with open(outfile, 'wb') as file: 
            pickle.dump(results, file)
        print ('got std uh oh')

        results['min'] = np.nanmin(X_on_disk)
        with open(outfile, 'wb') as file: 
            pickle.dump(results, file)
        print ('got min')

        results['max'] = np.nanmax(X_on_disk)
        with open(outfile, 'wb') as file: 
            pickle.dump(results, file)
        print ('got max')