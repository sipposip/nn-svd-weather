#! /proj/bolinc/users/x_sebsc/anaconda3/envs/nn-svd-env/bin/python

#SBATCH -A snic2019-1-2
#SBATCH --time=2-00:00:00
#SBATCH -N 1

# run on tetralith in /home/x_sebsc/nn_ensemble_nwp/plasimt42

import os
import sys
import pickle
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
import scipy.sparse.linalg
from dask.diagnostics import ProgressBar
ProgressBar().register()

modelname = 'plasimt42'
train_years = 100
i_train = 1 # ensemble member of net ensemble
outdir='output'
os.system(f'mkdir -p {outdir}')

ifile = '/proj/bolinc/users/x_sebsc/gcm_complexity_machinelearning/models/preprocessed/' + modelname + 'reordered.merged.nc'
# ifile='/climstorage/sebastian/gcm_complexity_machinelearning/modelruns/old/plasim_t21/plasimt21.reordered.normalized.merged.nc'
# ifile='/home/s/sebsc/pfs/nn_ensemble/plasimt42/modeldata/'+ modelname + 'reordered.merged.nc'

lead_time = 1  # days, lead time used for training

test_years = 1
train_years_offset = 0
days_per_year = 365  # noteL we are ignoring leap years in plasim here,

N_test = days_per_year * test_years
N_train = days_per_year * train_years

## parameters for the neural network

# fixed (not-tuned params)
batch_size = 32
num_epochs = 100
pool_size = 2
drop_prob = 0
conv_activation = 'relu'

## the params came out of the tuning process (for pumat21, Scher 2018)
params = {'conv_depth': 32, 'hidden_size': 500,
          'kernel_size': 6, 'lr': 0.0001, 'n_hidden_layers': 0}

param_string = modelname +'_'+ '_'.join([str(e) for e in (train_years,train_years_offset,num_epochs,lead_time, i_train)])

# mapping channel to variable name
target_var = 'zg500'
varnames = ['ua', 'va', 'ta', 'zg']
keys = [varname + str(lev) for varname in varnames for lev in range(100, 1001, 100)]
varname_to_levidx = {key: levidx for key, levidx in zip(keys, range(len(keys)))}
target_lev=varname_to_levidx[target_var]

data = xr.open_dataarray(ifile, chunks={'time':3600})  # we have to define chunks,
# then the data is opened as dask -array (out of core)

# convert to 32 bit
data = data.astype('float32')

# check that we have enough data for the specifications
if N_train + N_test > data.shape[0]:
    raise Exception('not enough timesteps in input file!')

# in the test-train split, the test data comes first. In the trianing,
# 30 years were skipped as test years, and then the training period began. This was a relict
# from the GMD paper, were this was necessary.from
# here, this mmeans we can simply use the start as test data, up to 30 years

data_test = data[:N_test]

# load the normalization weights to normalize the test data in the same way
# as the training data
norm_mean = xr.open_dataarray('data/norm_mean_'+param_string+'.nc')
norm_std = xr.open_dataarray('data/norm_std_'+param_string+'.nc')

data_test = (data_test - norm_mean) / norm_std
data_test = np.array(data_test, dtype='float32')
Nsamples, Nlat, Nlon,n_channels_in = data_test.shape


data = data_test
# now load the trained network
# since here we did the traning with an older version of keras,
# we cannot directly load the model, but we need to load the architecture
# and the weights separately.
## now load the trained network
weight_file = 'data/weights_'+param_string+'.h5'
architecture_file = 'data/modellayout_'+param_string+'.json'

model = tf.keras.models.model_from_json(json.load(open(architecture_file,'r')))
# load the weights form the training
model.load_weights(weight_file)


@tf.function  # this compiles the function into a tensorflow graph. does not change the behavior,
# but boosts performance by a factor of ~4 in our case
def net_jacobian(x, leadtime=1):
    """
        compute jacobian of network forecasts for leadtime timesteps.
        the output of the network is regridded to reduced resolution (defined by n_resample)
        This is to speed up the computation of the perturbed initial states
    """
    x = tf.convert_to_tensor(x[np.newaxis,...])

    with tf.GradientTape(persistent=True) as gt:
        gt.watch(x)
        pred = x
        for i in range(leadtime):
            pred = model(pred)

        if target_lev !='all':
            pred = tf.expand_dims(pred[:,:,:,target_lev], -1)

        pred = tf.image.resize(pred, size=(int(pred.shape[1]//n_resample),
                                           int(pred.shape[2]//n_resample)))

    J = gt.jacobian(pred, x, parallel_iterations=4, experimental_use_pfor=False)
    J = tf.squeeze(J)
    # returns dimension (Nlat,Nlon,Nlat,Nlon)
    return J


def create_pertubed_states(x, leadtime=1):

    L = net_jacobian(x, leadtime)
    # jacobian is format (n_output, n_input)
    if target_lev =='all':
        L = np.array(L).reshape((Nlat*Nlon*n_channels_in//(n_resample**2),Nlat*Nlon*n_channels_in))
    else:
        L = np.array(L).reshape((Nlat*Nlon//(n_resample**2),Nlat*Nlon*n_channels_in))
    # compute leading n_svs singular vectors
    u, s, vh  = scipy.sparse.linalg.svds(L,k=n_svs)
    svecs = vh
    assert(svecs.shape==(n_svs,Nlat*Nlon*n_channels_in))
    return svecs


#
tres_factor = 1
data = data[::tres_factor]
n_svs=10
n_resample = 1  # factor to reduce the output resolution/diension. 1 causes OOM on nvidia k80, but works on v100
svd_leadtime=4 # in multiples of lead_time * lead_time_hours. in our case, 4 is 96 hours (2days)

svd_params = f'n_svs{n_svs}_n_nresample{n_resample}_target_lev{target_lev}_svdleadtime{svd_leadtime}'

# note: here we compute the jacobian for all input states, also for those that we have to discard
# due to lack of truth data (last n_forecast_steps states)
x_init_svecs = np.array([create_pertubed_states(x, leadtime=svd_leadtime) for x in tqdm(data)])


np.save(f'{outdir}/jacobians_{param_string}_{test_years}_{svd_params}_tres{tres_factor}.npy',x_init_svecs)






