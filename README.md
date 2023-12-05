# Eikonal Tomography Using FNO Architecture

In seismology, accurately inferring subsurface velocity fields from observed travel times
is crucial for applications such as earthquake seismology and resource exploration. This
problem requires solving the Eikonal equation, which is a difficult task. Traditional methods
to solve the Eikonal equation, such as the Fast-Marching-Method (FMM), though effective,
are usually computationally expensive. 

We introduce an deep learning approach that uses
Fourier Neural Operators (FNOs) to find the travel times, which is significantly faster than
the FMM. We demonstrate the practicality of the FNO technique through by implementing
the model on a small dataset consisting of many source-receiver pairs one velocity field,
demonstrating the feasibility of the FNO technique to estimate sensitivity kernels. Then,
we train the FNO model on a significantly larger dataset comprising over 5000 velocity
fields, indicating its potential for broader applications. We also apply the generated kernels
to the inverse problem, in which we infer the velocity field given a set of travel times.
While further research is necessary to fully generalize the results, this study marks a
promising step forward for seismic tomography.

## Setup on HPC

This project is supported on Mac and Caltech HPC. The installation instructions will demonstrate only how to set up the codebase on HPC, which is Caltech's High Performance Computing clusters.

First, you will need to SSH into HPC, and you can login as follows:

```sh
$ ssh username@hpc.caltech.edu
[username@login1 ~] cd /central/groups/mlprojects/eikonal
[username@login1 eikonal] ls
```

You will be asked for your access password and Duo 2FA authentication.

Our setup uses VS Code's [Remote Explorer](https://code.visualstudio.com/docs/remote/ssh), where you should connect to hpc.caltech.edu at the folder `/central/groups/mlprojects/eikonal`. You can open a terminal there and run the commands in a terminal instead of using SSH directly like mentioned before. Remember that you will be asked for your access password first, and then should type 1 on the next prompt to be sent a Duo push for verification.

### Conda environment

If this is your first time working in HPC, you will need to run:
```sh
# Install Conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh ./Miniconda3-latest-Linux-x86_64.sh

# Set up conda using new installation
source ~/.bashrc

# Set up for jupyter notebook
conda install ipykernel
python -m ipykernel install --user --name base --display-name "Python (base)"
conda install -c anaconda jupyter
```

Then, to create the environment and enable, you should run:

```sh
$ conda env create -f Code/conda-envs/ejpatel_env.yml -n Eikonal
$ conda init bash
$ conda init zsh
$ source ~/.bashrc

(base) [username@login1 eikonal]$ conda activate Eikonal
(Eikonal) [username@login1 eikonal]$ 
```

Finally, to set up everything with some final libraries:

```sh
(Eikonal) [username@login1 eikonal]$ module load cuda/11.8
(Eikonal) [username@login1 eikonal]$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

You will need to activate the conda environment every time you connect via SSH. You can avoid this by doing:
```sh
(Eikonal) [username@login1 eikonal]$ echo "conda activate Eikonal" >> ~/.bashrc
```

This works because the `~/.bashrc` file (and all commands in it) is run when a shell session starts!

From here, you can run the following to test if the environment is working:
```sh
(Eikonal) [username@login1 eikonal]$ 
```

### Jupyter Notebooks

You can access HPC interactively [here](https://interactive.hpc.caltech.edu/), and navigating to `Interactive Apps > Jupyter Notebook - Compute Host`.
