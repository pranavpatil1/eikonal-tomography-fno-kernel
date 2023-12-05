#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH -J "eikonal"   # job name
#SBATCH --mail-user=ppatil@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL
#SBATCH -A mlprojects

# Load any modules you may need
module load cuda/11.8

# modes, width, batch size, epochs, save interval, add --small-dataset to use smaller dataset
python /groups/mlprojects/eikonal/Code/pranav_train_on_encoded.py -m 12 -w 128 -b 64 -e 1000 -s 10
# python /groups/mlprojects/eikonal/Code/pranav_visualize_model.py
