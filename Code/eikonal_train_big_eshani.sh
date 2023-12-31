#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH -J "eikonalBig"   # job name
#SBATCH -A mlprojects
#SBATCH --mail-user=ejpatel@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL

# Load any modules you may need
module load cuda/11.8

python /groups/mlprojects/eikonal/Code/ejpatel_big/eikonal_train_big_eshani.py 