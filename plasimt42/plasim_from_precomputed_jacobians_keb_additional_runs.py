#! /pfs/nobackup/home/s/sebsc/miniconda3/envs/tf2-env/bin/python

#SBATCH -A SNIC2019-3-611
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:k80:1

import os
import json
import matplotlib
import gc
import pickle
matplotlib.use('agg')
from pylab import plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import xarray as xr
from dask.diagnostics import ProgressBar
ProgressBar().register()

modelname = 'plasimt42'
train_years = 100
i_train = 1 # ensemble member of net ensemble
outdir='output'
os.system(f'mkdir -p {outdir}')

# ifile = '/proj/bolinc/users/x_sebsc/gcm_complexity_machinelearning/models/preprocessed/' + modelname + 'reordered.merged.nc'

ifile='/home/s/sebsc/pfs/nn_ensemble/plasimt42/modeldata/'+ modelname + 'reordered.merged.nc'

# jacobi_dir = '/proj/bolinc/users/x_sebsc/nn_ensemble_nwp/plasimt42/output/'
jacobi_dir = '/pfs/nobackup/home/s/sebsc/nn_ensemble/plasimt42/output/'
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

norm_targlev = norm_std[target_lev].values

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


tres_factor = 1
data_reduced = data[::tres_factor]
n_svs=10
n_resample = 1  # factor to reduce the output resolution/diension. 1 causes OOM on k80
svd_leadtime=2 # in multiples of lead_time * lead_time_hours. in our case, 2 is 48 hours (2days)

svd_params = f'n_svs{n_svs}_n_nresample{n_resample}_target_lev{target_lev}_svdleadtime{svd_leadtime}'

def create_pertubed_states_svd(x, svecs):

    assert(svecs[0].shape==(Nlat*Nlon*n_channels_in,))
    x_perts = []
    for i in range(n_ens//2):

        p = np.random.normal(size=n_svs)
        p[p>3]=3
        p[p<-3]=-3
        pert_scales = pert_scale * p
        pert = np.sum([pert_scales[ii] * svecs[ii] for ii in range(n_svs)], axis=0)
        # this is now flattened. reshape
        pert = pert.reshape((Nlat,Nlon,n_channels_in))
        # "negative" perturbation
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
    #assert(np.allclose(np.mean(x_pert-x, axis=0),0.))
    #assert(np.allclose(np.std(x_pert, axis=0),pert_scale))
    if not np.allclose(np.mean(x_pert-x, axis=0),0.):
        print(f'warning, np.mean(x_pert-x, axis=0)={np.mean(x_pert-x, axis=0)}')
    if not np.allclose(np.std(x_pert, axis=0),pert_scale):
        print(f'warning, np.allclose(np.std(x_pert, axis=0),pert_scale)={np.allclose(np.std(x_pert, axis=0),pert_scale)}')

    return x_pert

# prediction
target_max_leadtime = 14  # in contrast to the era5 script, here we work with days instead of hours
max_forecast_steps = target_max_leadtime // lead_time


def compute_mse(x,y):
    '''mse per sample/forecast'''
    assert(x.shape == y.shape)
    assert(x.shape[1:] == (Nlat,Nlon,n_channels_in))
    return np.mean((x[:,:,:,target_lev]-y[:,:,:,target_lev])**2,axis=(1,2)) * (norm_targlev**2)


def compute_mse_single_fc(x,y):
    '''mse single forecast'''
    assert(x.shape == y.shape)
    assert(x.shape == (Nlat,Nlon,n_channels_in))
    return np.mean((x[:,:,target_lev]-y[:,:,target_lev])**2,axis=(0,1)) * (norm_targlev**2)


svecs_all = np.load(f'{jacobi_dir}/jacobians_{param_string}_{test_years}_{svd_params}_tres{tres_factor}.npy')
assert(len(data_reduced) == len(svecs_all))


# network member ensemble
network_ensemble_all = []
n_ens_netens_max = 10
for imem in range(n_ens_netens_max):
    _param_string = modelname + '_' + '_'.join(
        [str(e) for e in (train_years, train_years_offset, num_epochs, lead_time, imem)])
    _architecture_file='data/modellayout_'+_param_string+'.json'
    _weight_file='data/weights_'+_param_string+'.h5'
    _model = tf.keras.models.model_from_json(json.load(open(_architecture_file, 'r')))
    # load the weights form the training
    _model.load_weights(_weight_file)
    network_ensemble_all.append(_model)


target_max_leadtime = 14  # in contrast to the era5 script, here we work with days instead of hours
max_forecast_steps = target_max_leadtime // lead_time
time_resolution_hours = 24 # only for plotting
x_init_ctrl = data_reduced[:-int(max_forecast_steps//tres_factor)]


# loop over different parameters for th eensemble.
# in each loop iteration, all forecasts are made and evaluated
for n_ens in [2,100]:

    if n_ens <=n_ens_netens_max:
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
            ensvar_mean_netens = np.mean(y_pred_ensvar_netens_2d[:,:,:,target_lev], axis=(1, 2)) * (norm_targlev**2)
            res_mse_netens.append(mse_ensmean_netens)
            res_ensvar_netens.append(ensvar_mean_netens)
            mse_ensmean_member = np.array([compute_mse(y_pred_netens[:,i], truth) for i in range(n_ens)])
            res_mse_netens_permember.append(mse_ensmean_member)

            for i in range(n_ens):
                y_pred_netens[:,i] = network_ensemble_all[i].predict(y_pred_netens[:,i])

        res_mse_netens = np.array(res_mse_netens)
        res_ensvar_netens = np.array(res_ensvar_netens)
        res_mse_netens_permember = np.array(res_mse_netens_permember)


    print(f'watch out when reading output data: for n_ens>n_ens_netens_max(={n_ens_netens_max}), simply the values for\
                n_ens=n_ens_netens will be reused for the netens!')
    # perturbed ensembles

    for pert_scale in [1, 3]:

        svd_params = f'n_svs{n_svs}_n_ens{n_ens}_pertscale{pert_scale}_nresample{n_resample}_svdleadtime{svd_leadtime}'
        # initialize, we have to discard the last max_leadtime initial fields, because we dont
        # have the targets for them. since we decreased the timeresolution, we have
        # to account for the decreased timeresolution as well
        x_init_ctrl = data_reduced[:-int(max_forecast_steps//tres_factor)]
        svecs = svecs_all[:-int(max_forecast_steps//tres_factor)]
        assert(len(x_init_ctrl) == len(svecs))
        n_samples = len(x_init_ctrl)

        res_mse_ctrl = np.zeros((max_forecast_steps, n_samples))
        res_mse_pers = np.zeros((max_forecast_steps, n_samples))
        res_mse_ensmean_svd = np.zeros((max_forecast_steps, n_samples))
        res_mse_ensmean_rand = np.zeros((max_forecast_steps, n_samples))
        leadtimes = np.zeros((max_forecast_steps,))
        res_mse_pers = np.zeros((max_forecast_steps, n_samples))
        res_ensvar_svd = np.zeros((max_forecast_steps, n_samples))
        res_ensvar_rand = np.zeros((max_forecast_steps, n_samples))

        # svd ensemble
        for i_init in tqdm(range(n_samples)):

            x_init_pert_svd = create_pertubed_states_svd(x_init_ctrl[i_init], svecs_all[i_init])
            x_init_pert_rand = create_pertubed_states_rand(x_init_ctrl[i_init])
            # x_init_pert_svd_flat = x_init_pert_svd.reshape((n_ens, Nlat, Nlon, n_channels_in))
            # y_pred_ens = x_init_pert_svd_flat
            y_pred_ens_svd = x_init_pert_svd
            y_pred_ens_rand = x_init_pert_rand
            y_pred_ctrl = x_init_ctrl[i_init][np.newaxis, ...]
            x_init_pers = x_init_ctrl[i_init][np.newaxis, ...]

            for ilead, fc_step in enumerate(range(0, max_forecast_steps)):
                leadtime = fc_step * (lead_time * time_resolution_hours)
                leadtimes[ilead] = leadtime
                # this leadtime is in [h], and used for plotting and labelling od the data
                # it is not to be confused with lead_time, which is in [timesteps] and which is
                # the leadtime the NN is trained on

                # we start with evaluation at leadtime 0 (so without prediction yet)
                y_pred_ensmean = np.mean(y_pred_ens_svd, axis=0)  # member dimension is 1st dim (0)
                y_pred_ensvar_2d = np.var(y_pred_ens_svd, axis=0)
                y_pred_ensmean_rand = np.mean(y_pred_ens_rand, axis=0)  # member dimension is 1st dim (0)
                y_pred_ensvar_2d_rand = np.var(y_pred_ens_rand, axis=0)

                # get right target data for this leadtime
                if fc_step < max_forecast_steps:
                    truth_all = data[fc_step * lead_time:-(max_forecast_steps - fc_step * lead_time):tres_factor]
                else:
                    truth_all = data[fc_step * lead_time::tres_factor]

                truth = truth_all[i_init]

                assert (truth.shape == y_pred_ctrl.squeeze().shape)  # y_pred_ctrl has an empty time dim
                mse_ctrl = compute_mse_single_fc(y_pred_ctrl.squeeze(), truth)
                mse_ensmean_svd = compute_mse_single_fc(y_pred_ensmean, truth)
                mse_ensmean_rand = compute_mse_single_fc(y_pred_ensmean_rand, truth)

                res_mse_ctrl[ilead, i_init] = mse_ctrl
                res_mse_ensmean_svd[ilead, i_init] = mse_ensmean_svd
                res_mse_ensmean_rand[ilead, i_init] = mse_ensmean_rand

                mse_pers = compute_mse_single_fc(x_init_pers.squeeze(), truth)
                res_mse_pers[ilead, i_init] = mse_pers
                ensvar_mean_svd = np.mean(y_pred_ensvar_2d[:, :, target_lev], axis=(0, 1)) * (norm_targlev ** 2)
                ensvar_mean_rand = np.mean(y_pred_ensvar_2d_rand[:, :, target_lev], axis=(0, 1)) * (norm_targlev ** 2)
                res_ensvar_rand[ilead, i_init] = ensvar_mean_rand
                res_ensvar_svd[ilead, i_init] = ensvar_mean_svd

                # predict next timestep
                y_pred_ctrl = model.predict(y_pred_ctrl, verbose=0)
                y_pred_ens_svd = model.predict(y_pred_ens_svd, verbose=0)
                y_pred_ens_rand = model.predict(y_pred_ens_rand, verbose=0)


        plt.figure()
        plt.plot(leadtimes, np.sqrt(np.mean(res_mse_ctrl,1)), label='rmse ctrl', color='black')
        plt.plot(leadtimes, np.sqrt(np.mean(res_mse_ensmean_svd,1)), label='rmse ensmean svd', color='#1b9e77')
        plt.plot(leadtimes, np.sqrt(np.mean(res_mse_ensmean_rand,1)), label='rmse ensmean rand', color='#7570b3')
        plt.plot(leadtimes, np.sqrt(np.mean(res_mse_netens,1)), label='rmse ensmean netens', color='#d95f02')
        plt.plot(leadtimes, np.sqrt(np.mean(res_mse_pers,1)), label='rmse persistence')
        plt.plot(leadtimes, np.sqrt(np.mean(res_ensvar_svd,1)), label='spread svd', color='#1b9e77', linestyle='--')
        plt.plot(leadtimes, np.sqrt(np.mean(res_ensvar_rand,1)), label='spread rand', color='#7570b3', linestyle='--')
        plt.plot(leadtimes, np.sqrt(np.mean(res_ensvar_netens,1)), label='spread netens', color='#d95f02', linestyle='--')
        for imem in range(res_mse_netens_permember.shape[1]):
            plt.plot(leadtimes, np.sqrt(np.mean(res_mse_netens_permember[:,imem],1)), label=f'{imem}')
        plt.legend()
        plt.savefig(f'{outdir}/leadtime_vs_skill_and_spread_{param_string}_{test_years}_{svd_params}.svg')

        corrs_svd = []
        corrs_rand = []
        corrs_netens = []
        for ltime in range(1,max_forecast_steps):
            corrs_svd.append(np.corrcoef(np.sqrt(res_ensvar_svd[ltime]).squeeze(), np.sqrt(res_mse_ensmean_svd[ltime]).squeeze())[0, 1])
            corrs_rand.append(np.corrcoef(np.sqrt(res_ensvar_rand[ltime]).squeeze(), np.sqrt(res_mse_ensmean_rand[ltime]).squeeze())[0, 1])
            corrs_netens.append(np.corrcoef(np.sqrt(res_ensvar_netens[ltime]).squeeze(), np.sqrt(res_mse_netens[ltime]).squeeze())[0, 1])

        plt.figure()
        plt.plot(leadtimes[1:], corrs_svd, label='svd', color='#1b9e77')
        plt.plot(leadtimes[1:], corrs_netens, label='netens', color='#d95f02')
        plt.plot(leadtimes[1:], corrs_rand, label='rand', color='#7570b3')
        plt.legend()
        plt.ylim((-0.2, 1))
        plt.xlabel('leadtime [h]')
        plt.ylabel('correlation')
        sns.despine()
        plt.title(f'n_ens:{n_ens}')
        plt.savefig(f'{outdir}/plasimt42_vs_error_spread_corr_netens_{param_string}_{test_years}_{svd_params}.svg')

        plt.close('all')

        df = pd.DataFrame({'leadtime':leadtimes,
                           'rmse_ctrl':np.sqrt(np.mean(res_mse_ctrl,1)).squeeze(),
                           'mse_ensmean_svd':np.sqrt(np.mean(res_mse_ensmean_svd,1)).squeeze(),
                           'mse_ensmean_rand':np.sqrt(np.mean(res_mse_ensmean_rand,1)).squeeze(),
                           'mse_ensmean_netens':np.sqrt(np.mean(res_mse_netens,1)).squeeze(),
                           'spread_svd':np.sqrt(np.mean(res_ensvar_svd,1)).squeeze(),
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
        df.to_csv(f'{outdir}/plasimt42_leadtime_vs_skill_and_spread_{param_string}_{test_years}_{svd_params}.csv')

        # save single forecast results as well

        out = {'leadtime':leadtimes,
               'rmse_ctrl':res_mse_ctrl,  # these arrays have shape (leadtime,samples)
               'mse_ensmean_svd':res_mse_ensmean_svd,
               'mse_ensmean_rand':res_mse_ensmean_rand,
               'mse_ensmean_netens':res_mse_netens,
               'spread_svd':res_ensvar_svd,
               'spread_rand':res_ensvar_rand,
               'spread_netens':res_ensvar_netens,
               'n_ens':n_ens,
               'pert_scale':pert_scale,
               }
        pickle.dump(out, open(f'{outdir}/plasimt42_leadtime_vs_skill_and_spread_{param_string}_{test_years}_{svd_params}.pkl','wb'))
