#! /pfs/nobackup/home/s/sebsc/miniconda3/envs/tf2-env/bin/python

#SBATCH -A SNIC2019-3-611
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:k80:1

"""

needs tensorflow 2.0

(source activate tf2-env on kebnekaise)
nn-svd-env on tetralith

run in kebnekaise
"""


import os
import pickle
import json
import threading
import matplotlib
matplotlib.use('agg')
from pylab import plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import tensorflow as tf
from tensorflow.keras.layers import Convolution2D,Dropout
from tensorflow.keras import layers
import tensorflow.keras as keras
import dask
from tqdm import tqdm

from dask.diagnostics import ProgressBar

ProgressBar().register()
# set a fixed threadpool for dask. I am not sure whether this is a good idea.
# the keras fit_generator spawns multile threads that load data via the generator,
# but the generator uses dask. on tetralith, net setting a fixed threadpool for dask
# lead to a very huge number of threads after some time....

# dask.config.set(scheduler="synchronous")


datadir='data'
outdir='output'
os.system(f'mkdir -p {datadir}')
os.system(f'mkdir -p {outdir}')

# basepath for input data
if 'NSC_RESOURCE_NAME' in os.environ and os.environ['NSC_RESOURCE_NAME']=='tetralith':
    basepath = '/proj/bolinc/users/x_sebsc/weather-benchmark/2.5deg/'
    norm_weights_folder = '/proj/bolinc/users/x_sebsc/nn_/benchmark/normalization_weights/'
else:
    basepath = '/home/s/sebsc/pfs/weather_benchmark/2.5deg/'
    norm_weights_folder = '/home/s/sebsc/pfs/nn_ensemble/era5/normalization_weights/'


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# K.set_session(tf.Session(config=config))


lead_time = 1   # in steps
N_gpu=0
load_data_lazily = True
modelname='era5_2.5deg_weynetal'
train_startyear=1979
train_endyear=2016
test_startyear = 2017
test_endyear = 2018
time_resolution_hours = 6
variables = ['geopotential_500']
invariants = ['lsm', 'z']
valid_split = 0.02

norm_weights_filenamebase = f'{norm_weights_folder}/normalization_weights_era5_2.5deg_{train_startyear}-{train_endyear}_tres{time_resolution_hours}_' \
    + '_'.join([str(e) for e in variables])


## parameters for the neural network

# fixed (not-tuned params)
batch_size = 32
num_epochs = 200
drop_prob=0


def read_e5_data(startyear, endyear,
                 variables='all'):

    all_variables = ['2m_temperature', 'mean_sea_level_pressure',
                     '10m_u_component_of_wind', '10m_v_component_of_wind']



    years = list(range(startyear, endyear+1))
    if variables=='all':
        variables = all_variables

    combined_ds = []
    for variable in variables:
        ifiles = [f'{basepath}/{variable}/{variable}_{year}_2.5deg.nc' for year in years]
        ds = xr.open_mfdataset(ifiles, chunks={'time':1}) # we need to chunk time by 1 to get efficient
        # reading of data whenrequesting lower time-resolution
        # this is now a dataset. we want to have a dataarray
        da = ds.to_array()
        # now the first dimension is the variable dimension, which should have length 1 (because we have only
        # 1 variable per file
        assert(da.shape[0]==1)
        # remove this dimension
        da = da.squeeze()
        if not load_data_lazily:
            da.load()
        combined_ds.append(da)


    return combined_ds


# lazily load the whole dataset
ds_whole = read_e5_data(test_startyear, test_endyear, variables=variables)
# this is now a lazy dask array. do not do any operations on this array outside the data generator below.
# if we do operations before, it will severly slow down the data loading throughout the training.

# load normalization weights
norm_mean = xr.open_dataarray(norm_weights_filenamebase+'_mean.nc').values
norm_std = xr.open_dataarray(norm_weights_filenamebase+'_std.nc').values


n_data = ds_whole[0].shape[0]
N_train =  n_data// time_resolution_hours
n_valid = int(N_train * valid_split)

Nlat,Nlon,=ds_whole[0].shape[1:3]

Nlat = Nlat//2 # only NH

n_channels_out = len(variables)


n_channels_in = n_channels_out



param_string = f'{modelname}_{train_startyear}-{train_endyear}'


time_indices_all = np.arange(0,n_data-time_resolution_hours*lead_time,time_resolution_hours)
data_train = np.array(ds_whole[0][time_indices_all], dtype='float32')
data_train = (data_train - norm_mean)/norm_std

x_train = data_train[:-lead_time]
y_train = data_train[lead_time:]

# add (empty) channel dimension
x_train = np.expand_dims(x_train,axis=-1)
y_train = np.expand_dims(y_train,axis=-1)
x_train = x_train[:,-Nlat:]
y_train = y_train[:,-Nlat:]


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


#modelfile='data/trained_model_era5_2.5deg_weynetal_1979-2016.h5' # this net has higher skill than the 1-10 ones (dont know why)
modelfile='data_mem2/trained_model_era5_2.5deg_weynetal_1979-2016_mem2.h5' # this net has higher skill than the 1-10 ones (dont know why)
model = keras.models.load_model(modelfile,
    custom_objects={'PeriodicPadding': PeriodicPadding})

print(model.summary())



lat = ds_whole[0].lat.values[-Nlat:]


data = data_train[:,-Nlat:]
if n_channels_in == 1:
    data = np.expand_dims(data, -1)



# reduce to 2-daily resolution
tres_factor = 8
data_reduced = data[::tres_factor]
n_svs=10
n_ens=10
pert_scale=1.0 # only for filename of jacobian
n_resample = 1  # factor to reduce the output resolution/diension
svd_leadtime=8 # in multiples of lead_time * lead_time_hours. in our case, 8 is 48 hours (2days)

svd_params = f'n_svs{n_svs}_n_ens{n_ens}_pertscale{pert_scale}_nresample{n_resample}_svdleadtime{svd_leadtime}'

def create_pertubed_states_svd(x, svecs):

    assert(svecs[0].shape==(Nlat*Nlon*n_channels_in,))
    x_perts = []
    for i in range(n_ens//2):

        # get sample of size n_svs from truncated gaussian distribution (truncated at 3)
        p = np.random.normal(size=n_svs)
        p[p>3]=3
        p[p<-3]=-3
        # now convert to distribution with desired scale
        pert_scales = pert_scale * p
        pert = np.sum([pert_scales[ii] * svecs[ii] for ii in range(n_svs)], axis=0)
        # this is now flattened. reshape
        pert = pert.reshape((Nlat,Nlon,n_channels_in))
        # "positive" and "negative" perturbation
        x_pert1 = x + pert
        x_pert2 = x - pert

        x_perts.append(x_pert1)
        x_perts.append(x_pert2)

    x_perts = np.array(x_perts)
    return x_perts


def create_pertubed_states_rand(x):
    perts = np.random.normal(0, scale=pert_scale, size=(n_ens,Nlat,Nlon,n_channels_in))
    # since this is a finite sample from the distribution,
    # it does not necessarily have 0 mean and pert_scale variance (section 4.5 in Tellus paper)
    # and we therefore rescale them for every channel and gridpoint
    perts = (perts - perts.mean(axis=0)) / perts.std(axis=0) * pert_scale
    # perts is (n_ens,N), and x is (N), so we can use standard numpy broadcasting
    # to create n_ens states
    x_pert = x + perts
    assert(np.allclose(np.mean(x_pert-x, axis=0),0.))
    assert(np.allclose(np.std(x_pert, axis=0),pert_scale))
    return x_pert

# prediction
target_max_leadtime = 5*24
max_forecast_steps = target_max_leadtime // (lead_time*time_resolution_hours)



def compute_mse(x,y):
    '''mse per sample/forecast'''
    assert(x.shape == y.shape)
    assert(x.shape[1:] == (Nlat,Nlon,n_channels_in))
    return np.mean((x-y)**2,axis=(1,2)) * norm_std**2


# load precomputed singular vectors
svecs_all =np.load(f'{outdir}/jacobians_{param_string}_{test_startyear}-{test_endyear}_{svd_params}_tres{tres_factor}.npy')
assert(len(data_reduced) == len(svecs_all))


# network member ensemble
member_selection_scores = np.load('mse_per_mem.npy')
# select only the 20 best members (at lead time index 10)
best_mem_idcs = member_selection_scores[10,:].argsort()[:20][::-1]
best_mem_idcs = best_mem_idcs +1
network_ensemble_all = []
for imem in best_mem_idcs:
    _modelfile=f'data_mem{imem}/trained_model_era5_2.5deg_weynetal_1979-2016_mem{imem}.h5'
    _model = keras.models.load_model(_modelfile,    custom_objects={'PeriodicPadding': PeriodicPadding})
    network_ensemble_all.append(_model)



x_init_ctrl = data_reduced[:-int(max_forecast_steps//tres_factor)]



# loop over different parameters for th eensemble.
# in each loop iteration, all forecasts are made and evaluated
for n_ens in [2,4,10,20,100,1000]:

    if n_ens <=20:
        # network member ensemble

        # initialize (via repeating x_init_ctrl)
        y_pred_netens = np.zeros((len(x_init_ctrl), n_ens, Nlat, Nlon, n_channels_in))
        for i in range(n_ens):
            y_pred_netens[:,i] = x_init_ctrl

        res_mse_netens = []
        res_ensvar_netens = []

        res_mse_netens_permember = []

        for ilead, fc_step in enumerate(range(0, max_forecast_steps)):
            print(ilead)

            y_pred_ensmean_netens = np.mean(y_pred_netens, axis=1)
            y_pred_ensvar_netens_2d = np.var(y_pred_netens, axis=1)

            # get right target data for this leadtime
            if fc_step < max_forecast_steps:
                truth = data[fc_step * lead_time:-(max_forecast_steps - fc_step * lead_time):tres_factor]
            else:
                truth = data[fc_step * lead_time::tres_factor]

            mse_ensmean_netens = compute_mse(y_pred_ensmean_netens, truth)
            ensvar_mean_netens = np.mean(y_pred_ensvar_netens_2d, axis=(1, 2)) * norm_std ** 2
            res_mse_netens.append(mse_ensmean_netens)
            res_ensvar_netens.append(ensvar_mean_netens)
            mse_ensmean_member = np.array([compute_mse(y_pred_netens[:,i], truth) for i in range(n_ens)])
            res_mse_netens_permember.append(mse_ensmean_member)

            for i in range(n_ens):
                y_pred_netens[:,i] = network_ensemble_all[i].predict(y_pred_netens[:,i])

        res_mse_netens = np.array(res_mse_netens)
        res_ensvar_netens = np.array(res_ensvar_netens)
        res_mse_netens_permember = np.array(res_mse_netens_permember)

    print('watch out when reading output data: for n_ens>20, simply the values for n_ens=20 will be used for the netens!')
    # perturbed ensembles

    for pert_scale in [0.001,0.003,0.01, 0.03,0.1,0.3,1, 3]:

        svd_params = f'n_svs{n_svs}_n_ens{n_ens}_pertscale{pert_scale}_nresample{n_resample}_svdleadtime{svd_leadtime}'
        # initialize, we have to discard the last max_leadtime initial fields, because we dont
        # have the targets for them. since we decreased the timeresolution, we have
        # to account for the decreased timeresolution as well
        x_init_ctrl = data_reduced[:-int(max_forecast_steps//tres_factor)]
        svecs = svecs_all[:-int(max_forecast_steps//tres_factor)]
        assert(len(x_init_ctrl) == len(svecs))
        # crate_perturbed_states_batch does not work with large batches (memory problems),
        # therefore for now we loop over all state and use the function for single states
        x_init_pert_svd = np.array([create_pertubed_states_svd(x_init_ctrl[i], svecs_all[i]) for i in tqdm(range(len(x_init_ctrl)))])
        x_init_pert_rand = np.array([create_pertubed_states_rand(x_init_ctrl[i]) for i in tqdm(range(len(x_init_ctrl)))])
        n_samples = len(x_init_ctrl)
        x_init_pers = x_init_ctrl.copy()
        y_pred_ctrl = x_init_ctrl

        # x_init_pert has shape (ntime,nens,Nlat,Nlon,Nchannel_in)
        x_init_pert_svd_flat = x_init_pert_svd.reshape((n_samples*n_ens,Nlat,Nlon,n_channels_in))
        y_pred_ens_svd_flat = x_init_pert_svd_flat
        x_init_pert_rand_flat = x_init_pert_rand.reshape((n_samples*n_ens,Nlat,Nlon,n_channels_in))
        y_pred_ens_rand_flat = x_init_pert_rand_flat


        res_mse_ctrl = []
        res_mse_pers = []
        res_mse_ensmean_svd = []
        res_mse_ensmean_rand = []
        leadtimes = []
        res_mse_pers = []
        res_ensvar = []
        res_ensvar_rand = []

        for ilead,fc_step in enumerate(range(0,max_forecast_steps)):
            print(ilead)
            leadtime = fc_step * (lead_time*time_resolution_hours)
            leadtimes.append(leadtime)

            # we start with evaluation at leadtime 0 (so without prediction yet)
            y_pred_ens = y_pred_ens_svd_flat.reshape(x_init_pert_svd.shape)
            y_pred_ensmean = np.mean(y_pred_ens, axis=1)  # member dimension is 2nd dime
            y_pred_ensvar_2d = np.var(y_pred_ens, axis=1)
            y_pred_ens_rand = y_pred_ens_rand_flat.reshape(x_init_pert_rand.shape)
            y_pred_ensmean_rand = np.mean(y_pred_ens_rand, axis=1)  # member dimension is 2nd dime
            y_pred_ensvar_rand_2d = np.var(y_pred_ens_rand, axis=1)

            # get right target data for this leadtime
            if fc_step < max_forecast_steps:
                truth = data[fc_step*lead_time:-(max_forecast_steps-fc_step*lead_time):tres_factor]
            else:
                truth = data[fc_step*lead_time::tres_factor]

            assert(truth.shape == y_pred_ctrl.shape)
            mse_ctrl = compute_mse(y_pred_ctrl, truth)
            mse_ensmean_svd = compute_mse(y_pred_ensmean, truth)
            mse_ensmean_rand = compute_mse(y_pred_ensmean_rand, truth)
            res_mse_ctrl.append(mse_ctrl)
            res_mse_ensmean_svd.append(mse_ensmean_svd)
            res_mse_ensmean_rand.append(mse_ensmean_rand)
            mse_pers = compute_mse(x_init_pers, truth)
            res_mse_pers.append(mse_pers)
            ensvar_mean = np.mean(y_pred_ensvar_2d,axis=(1,2)) * norm_std **2
            res_ensvar.append(ensvar_mean)
            ensvar_mean_rand = np.mean(y_pred_ensvar_rand_2d, axis=(1, 2)) * norm_std ** 2
            res_ensvar_rand.append(ensvar_mean_rand)

            # predict next timestep
            y_pred_ctrl = model.predict(y_pred_ctrl, verbose=0)
            y_pred_ens_svd_flat = model.predict(y_pred_ens_svd_flat, verbose=0)
            y_pred_ens_rand_flat = model.predict(y_pred_ens_rand_flat, verbose=0)

        res_mse_ctrl = np.array(res_mse_ctrl)
        res_mse_ensmean_svd = np.array(res_mse_ensmean_svd)
        res_mse_ensmean_rand = np.array(res_mse_ensmean_rand)
        res_mse_pers = np.array(res_mse_pers)
        res_ensvar = np.array(res_ensvar)
        res_ensvar_rand = np.array(res_ensvar_rand)
        plt.figure()
        plt.plot(leadtimes, np.sqrt(np.mean(res_mse_ctrl,1)), label='rmse ctrl', color='black')
        plt.plot(leadtimes, np.sqrt(np.mean(res_mse_ensmean_svd,1)), label='rmse ensmean svd', color='#1b9e77')
        plt.plot(leadtimes, np.sqrt(np.mean(res_mse_ensmean_rand,1)), label='rmse ensmean rand', color='#7570b3')
        plt.plot(leadtimes, np.sqrt(np.mean(res_mse_netens,1)), label='rmse ensmean netens', color='#d95f02')
        plt.plot(leadtimes, np.sqrt(np.mean(res_mse_pers,1)), label='rmse persistence')
        plt.plot(leadtimes, np.sqrt(np.mean(res_ensvar,1)), label='spread svd', color='#1b9e77', linestyle='--')
        plt.plot(leadtimes, np.sqrt(np.mean(res_ensvar_rand,1)), label='spread rand', color='#7570b3', linestyle='--')
        plt.plot(leadtimes, np.sqrt(np.mean(res_ensvar_netens,1)), label='spread netens', color='#d95f02', linestyle='--')
        for imem in range(res_mse_netens_permember.shape[1]):
            plt.plot(leadtimes, np.sqrt(np.mean(res_mse_netens_permember[:,imem],1)), label=f'{imem}')
        plt.legend()
        plt.savefig(f'{outdir}/leadtime_vs_skill_and_spread_{param_string}_{test_startyear}-{test_endyear}_{svd_params}.svg')

        corrs_svd = []
        corrs_rand = []
        corrs_netens = []
        for ltime in range(1,max_forecast_steps):
            corrs_svd.append(np.corrcoef(np.sqrt(res_ensvar[ltime]).squeeze(), np.sqrt(res_mse_ensmean_svd[ltime]).squeeze())[0, 1])
            corrs_rand.append(np.corrcoef(np.sqrt(res_ensvar_rand[ltime]).squeeze(), np.sqrt(res_mse_ensmean_rand[ltime]).squeeze())[0, 1])
            corrs_netens.append(np.corrcoef(np.sqrt(res_ensvar_netens[ltime]).squeeze(), np.sqrt(res_mse_netens[ltime]).squeeze())[0, 1])

        plt.figure()
        plt.plot(leadtimes[1:], corrs_svd, label='svd', color='#1b9e77')
        plt.plot(leadtimes[1:], corrs_netens, label='netens', color='#d95f02')
        plt.plot(leadtimes[1:], corrs_rand, label='rand', color='#7570b3')
        plt.legend()
        plt.ylim((-0.2, 1))
        plt.xlabel('leadtime [MTU]')
        plt.ylabel('correlation')
        sns.despine()
        plt.title(f'n_ens:{n_ens}')
        plt.savefig(f'{outdir}/leadtime_vs_error_spread_corr_netens_{param_string}_{test_startyear}-{test_endyear}_{svd_params}.svg')

        plt.close('all')

        df = pd.DataFrame({'leadtime':leadtimes,
                           'rmse_ctrl':np.sqrt(np.mean(res_mse_ctrl,1)).squeeze(),
                           'mse_ensmean_svd':np.sqrt(np.mean(res_mse_ensmean_svd,1)).squeeze(),
                           'mse_ensmean_rand':np.sqrt(np.mean(res_mse_ensmean_rand,1)).squeeze(),
                           'mse_ensmean_netens':np.sqrt(np.mean(res_mse_netens,1)).squeeze(),
                           'spread_svd':np.sqrt(np.mean(res_ensvar,1)).squeeze(),
                           'spread_rand':np.sqrt(np.mean(res_ensvar_rand,1)).squeeze(),
                           'spread_netens':np.sqrt(np.mean(res_ensvar_netens,1)).squeeze(),
                           # the corrs arrays dont have an entry for leadtome=0, we we add 0 here to
                           # have the same length
                           'corr_svd':[0] + corrs_svd,
                           'corr_rand':[0] + corrs_rand,
                           'corr_netens':[0] + corrs_netens,
                           'n_ens':n_ens,
                           'pert_scale':pert_scale,
                           })
        df.to_csv(f'{outdir}/era5_leadtime_vs_skill_and_spread_{param_string}_{test_startyear}-{test_endyear}_{svd_params}.csv')
