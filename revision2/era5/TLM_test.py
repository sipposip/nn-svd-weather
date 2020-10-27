#! /proj/bolinc/users/x_sebsc/anaconda3/envs/nn-svd-env/bin/python
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -A snic2020-1-31

"""
needs tensorflow 2.0
run on tetralith
"""

import os
import sys
from pylab import plt
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
from dask.diagnostics import ProgressBar

ProgressBar().register()

datadir = 'data'

os.system(f'mkdir -p {datadir}')


# basepath for input data
if 'NSC_RESOURCE_NAME' in os.environ and os.environ['NSC_RESOURCE_NAME'] == 'tetralith':
    basepath = '/proj/bolinc/users/x_sebsc/weather-benchmark/2.5deg/'
    norm_weights_folder = '/proj/bolinc/users/x_sebsc/nn_ensemble_nwp/era5/normalization_weights/'
    outdir='/proj/bolinc/users/x_sebsc/nn_ensemble_nwp/era5/output/'
else:
    basepath = '/home/s/sebsc/pfs/weather_benchmark/2.5deg/'
    norm_weights_folder = '/home/s/sebsc/pfs/nn_ensemble/era5/normalization_weights/'
    outdir = 'output/'

lead_time = 1  # in steps
N_gpu = 0
load_data_lazily = True
modelname = 'era5_2.5deg_weynetal'
train_startyear = 1979
train_endyear = 2016
test_startyear = 2017
test_endyear = 2018
time_resolution_hours = 6
variables = ['geopotential_500']
invariants = ['lsm', 'z']
valid_split = 0.02

n_svs = 100
n_ens = 10  # this was only used for filename of jacobian
pert_scale = 1.0  # only for filename of jacobian
n_resample = 1  # factor to reduce the output resolution/diension
svd_leadtime = 4 #4*6=24h

norm_weights_filenamebase = f'{norm_weights_folder}/normalization_weights_era5_2.5deg_{train_startyear}-{train_endyear}_tres{time_resolution_hours}_' \
                            + '_'.join([str(e) for e in variables])

## parameters for the neural network

# fixed (not-tuned params)
batch_size = 32
num_epochs = 200
drop_prob = 0



param_string = f'{modelname}_{train_startyear}-{train_endyear}'

class PeriodicPadding(keras.layers.Layer):
    def __init__(self, axis, padding, **kwargs):
        """
        layer with periodic padding for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along
        padding: number of cells to pad
        """

        super(PeriodicPadding, self).__init__(**kwargs)

        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(padding, int):
            padding = (padding,)

        self.axis = axis
        self.padding = padding

    def build(self, input_shape):
        super(PeriodicPadding, self).build(input_shape)

    # in order to be able to load the saved model we need to define
    # get_config
    def get_config(self):
        config = {
            'axis': self.axis,
            'padding': self.padding,

        }
        base_config = super(PeriodicPadding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, input):

        tensor = input
        for ax, p in zip(self.axis, self.padding):
            # create a slice object that selects everything form all axes,
            # except only 0:p for the specified for right, and -p: for left
            ndim = len(tensor.shape)
            ind_right = [slice(-p, None) if i == ax else slice(None) for i in range(ndim)]
            ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
            right = tensor[ind_right]
            left = tensor[ind_left]
            middle = tensor
            tensor = tf.concat([right, middle, left], axis=ax)
        return tensor

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        for ax, p in zip(self.axis, self.padding):
            output_shape[ax] += 2 * p
        return tuple(output_shape)


# as main net we select net2
modelfile = 'data_mem2/trained_model_era5_2.5deg_weynetal_1979-2016_mem2.h5'
model = keras.models.load_model(modelfile,
                                custom_objects={'PeriodicPadding': PeriodicPadding})

print(model.summary())

tres_factor = 8

svd_params = f'n_svs{n_svs}_n_ens{n_ens}_pertscale{pert_scale}_nresample{n_resample}_svdleadtime{svd_leadtime}'



# prediction
target_max_leadtime = 5 * 24
max_forecast_steps = target_max_leadtime // (lead_time * time_resolution_hours)


# load precomputed singular vectors
svecs_all = np.load(
    f'{outdir}/jacobians_{param_string}_{test_startyear}-{test_endyear}_{svd_params}_tres{tres_factor}.npy')



# load precomputed jacobinas
pert_scales = np.linspace(-1,1,500)

# there is a small error in the script that wrote out the jacobians.
# instead of saving the initial condition for each jacobian, it saved all initial conditions
# therefore we just need to read one of them
data = np.load(f'{outdir}/jacobian_{svd_params}_0001_corresponding_init_cond.npy')
Nlat,Nlon,n_channels_in = data.shape[1:]

res_all = {'tlm':[],'finitdiff':[], 'pert_scales':pert_scales}
for i in tqdm(range(len(data))):
    svecs = svecs_all[i]
    J = np.load(f'{outdir}/jacobian_{svd_params}_{i:04d}.npy')
    x_init = data[i]

    #choose first svec
    svec = svecs[0]
    res_finitdiff = []
    res_tlm = []

    for pert_scale in tqdm(pert_scales):
        # make normal and perturbed forecast
        x_pert = x_init + pert_scale*svec.reshape(Nlat,Nlon,n_channels_in)
        fc_base = model.predict(np.expand_dims(x_init,0))
        fc_pert =  model.predict(np.expand_dims(x_pert,0))
        fc_diff = fc_pert - fc_base
        fc_diff_flat = fc_diff.flatten()
        # make "forecast" with jacobian
        fc_tlm = np.dot(J,pert_scale*svec)
        # compute area mean perturbation
        res_finitdiff.append(fc_diff_flat.mean())
        res_tlm.append(fc_tlm.mean())

    res_all['finitdiff'].append(res_finitdiff)
    res_all['tlm'].append(res_tlm)

pickle.dump(res_all,open('tlm_test_result.pkl','wb'))
