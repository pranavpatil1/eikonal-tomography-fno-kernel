#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=12:00:00   # walltime
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
# conda activate ejpatel_env

python /groups/mlprojects/eikonal/Code/fno_training_small_pranav.py 