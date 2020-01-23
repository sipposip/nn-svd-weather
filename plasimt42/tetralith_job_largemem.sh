#! /bin/bash

#SBATCH -A snic2019-1-2
#SBATCH -N 1

# ${model} ${train_years} ${train_years_offset} ${load_lazily} must be passed in by sbatch
set -u
# the original training script does not work the tensorflow v2, because v2 cannot deal with xarray/dask data
# as input to the fit method. therefore we use the environment from the GMD paper
/proj/bolinc/users/x_sebsc/anaconda3/envs/largescale-ML/bin/python train_network_puma_plasim_100epochlargemem.py ${model} ${train_years} ${i_train} ${load_lazily}

