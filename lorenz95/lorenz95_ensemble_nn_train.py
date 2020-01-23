#! /home/sebastian/anaconda3/envs/nn-svd-env/bin/python
# run on misu160
"""

needs tensorflow >=2.0

"""
import os
import matplotlib
matplotlib.use('agg')

import pickle
from tqdm import trange, tqdm
import pandas as pd
import numpy as np
from pylab import plt
import seaborn as sns
from scipy.integrate import odeint
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.parallel_for.gradients import jacobian
# set maximum numpber of CPUs to use
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16,
                       allow_soft_placement=True)
session = tf.compat.v1.Session(config=config)



# paramters for experiments
N = 200  # number of variables
F = 8
Nsteps = 10000
Nspinup = 500
Nsteps = Nsteps + Nspinup
tstep=0.01
t_arr = np.arange(0, Nsteps) * tstep
lead_time = 10
# fixed params neural network
n_epoch = 30
noise = 0.1
n_test = 500
noise_on_traindata = True
noise_on_testdata = True
svd_leadtime= 4 #https://www.tandfonline.com/doi/pdf/10.1111/j.1600-0870.2006.00197.x?needAccess=true use 0.4 timeunits
                # here we define it in multiples of the network timestep (which is 10 times the model timestep of 0.01
                #, so one means 0.1
n_svs = 10
outdir = 'data'

os.system(f'mkdir -p {outdir}')

def lorenz96(x,t):

  # compute state derivatives
  d = np.zeros(N)
  # first the 3 edge cases: i=1,2,N
  d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
  d[1] = (x[2] - x[N-1]) * x[0]- x[1]
  d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
  d[2:N-1] =  (x[2+1:N-1+1] - x[2-2:N-1-2]) * x[2-1:N-1-1] - x[2:N-1]  ## this is equivalent but faster
  # add the forcing term
  d = d + F

  return d

# we make two runs, started with slightly different initial conditions
# one will be the training and one the test run
x_init1 = F*np.ones(N) # initial state (equilibrium)
x_init1[19] += 0.01 # add small perturbation to 20th variable

modelrun_train = odeint(lorenz96,y0=x_init1, t= t_arr)

x_init2 = F*np.ones(N)
x_init2[1] += 0.05 #

modelrun_test = odeint(lorenz96,y0=x_init2, t= t_arr)

# remove spinuo
modelrun_train = modelrun_train[Nspinup:]
modelrun_test = modelrun_test[Nspinup:]

# add "observation" noise
if noise_on_traindata:
    modelrun_train = modelrun_train + np.random.normal(scale=noise, size=modelrun_train.shape)
if noise_on_testdata:
    modelrun_test = modelrun_test + np.random.normal(scale=noise, size=modelrun_test.shape)


# for loezn96, we dont have to normalize per variable, because all should have the same
# st and mean anywary, so we compute the total mean,  and the std for each gridpoint and then
# average all std
norm_mean = modelrun_train.mean()
norm_std = modelrun_train.std(axis=0).mean()
modelrun_train = (modelrun_train  - norm_mean) / norm_std
modelrun_test = (modelrun_test - norm_mean) / norm_std



modelrun_test = modelrun_test.astype('float32')
modelrun_train = modelrun_train.astype('float32')

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

def train_network(X_train, y_train, lr,kernel_size, conv_depth, n_conv, activation):
    """
    :param X_train:
    :param y_train:
    :param kernel_size:
    :param conv_depth:
    :param n_conv: >=1
    :return:
    """

    n_channel = 1 # empty
    n_pad = int(np.floor(kernel_size/2))
    layers = [
        PeriodicPadding(axis=1,padding=n_pad,input_shape=(N ,n_channel)),
        keras.layers.Conv1D(kernel_size=kernel_size,filters=conv_depth, activation=activation,
                            padding='valid')]

    for i in range(n_conv-1):
        layers.append(PeriodicPadding(axis=1,padding=n_pad))
        layers.append(keras.layers.Conv1D(kernel_size=kernel_size,filters=conv_depth, activation=activation,
                            padding='valid'))

    layers.append(PeriodicPadding(axis=1, padding=n_pad))
    layers.append(keras.layers.Conv1D(kernel_size=kernel_size, filters=1,  activation='linear', padding='valid'))


    model = keras.Sequential(layers)

    #  we have to add an empty channel dimension
    y_train = y_train[..., np.newaxis]
    X_train = X_train[..., np.newaxis]
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    print(model.summary())
    hist = model.fit(X_train, y_train, epochs=n_epoch, verbose=1, validation_split=0.1 ,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0,
                                                              patience=4,

                                                              verbose=1, mode='auto')]
                     )


    return model, hist


params = {"activation": "sigmoid", "conv_depth": 32, "kernel_size": 5, "lr": 0.003, "n_conv": 9}


X_train = modelrun_train[:-lead_time]
y_train = modelrun_train[lead_time:]

network, hist =  train_network(X_train, y_train, **params)


n_ens_list = [2,4,10,20, 100]
n_ens_max = max(n_ens_list)

# network ensemble
net_ens_all = []
for i in range(n_ens_max):
    print(f'training network number {i+1}')
    _network, hist =  train_network(X_train, y_train, **params)
    net_ens_all.append(_network)



## gradients


@tf.function  # this compiles the function into a tensorflow graph. does not change the behavior,
# but boosts performance by a factor of ~4 in our case
def net_jacobian(x, leadtime=1):
    x = tf.convert_to_tensor(x[np.newaxis,:,np.newaxis])
    # tuning considerations: parallel_iterations in GradientTape.jacobian,
    # and using persistent=True or False in GradientTape
    # parallel_iterations only works when persistent=True (otherwise it crashes), but
    # then it is actually slower than without parallel_iterations

    # Note: performance considerations: the function computes the jacobian only for a single input
    # this makes sense in a "operational forecast" setting. but in training/evaluation etc,
    # we could also use GradientTape.batch_jacobian instead, as this is much faster than
    # calling jacobian on many individuals samples.
    with tf.GradientTape(persistent=True) as gt:
      gt.watch(x)
      pred = x
      for i in range(leadtime):
          pred = network(pred)

    J = gt.jacobian(pred, x, parallel_iterations=4, experimental_use_pfor=False)
    J = tf.squeeze(J)
    return J




def create_pertubed_states(x, leadtime=1):

    L = net_jacobian(x, leadtime)
    L = np.array(L)
    u, s, vh = np.linalg.svd(L)
    evecs = vh
    # according to numpy documentation:
    # the rows of vh are the eigenvectors of L*L, and the columns of u are
    # the eigenvectors of LL*
    # s contains the eigenvalues in rising order
    # # we want the eigenvectors of L*L (2.7 in Palmer et al, https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.1994.0105)

    # select SVs. https://www.tandfonline.com/doi/pdf/10.1111/j.1600-0870.2006.00197.x?needAccess=true select 10,
    # but not always the first 10 (they use an exclution algorithm from https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.1994.0105)

    # here for a start, we simply select the first 10 (defined by n_svs
    svecs = evecs[:n_svs]

    x_perts = []
    for i in range(n_ens//2):

        p = np.random.normal(size=n_svs)
        p[p>3]=3
        p[p<-3]=-3
        pert_scales = pert_scale * p
        pert = np.sum([pert_scales[ii] * svecs[ii] for ii in range(n_svs)])
        # "negative" pertturbation
        x_pert1 = x + pert
        x_pert2 = x - pert

        x_perts.append(x_pert1)
        x_perts.append(x_pert2)

    x_perts = np.array(x_perts)
    return x_perts



sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
leadtimes = np.arange(int(200//lead_time)+1) * lead_time * tstep # only needed for plots
res_all = pd.DataFrame()
for n_ens in n_ens_list:

    # network ensemble
    net_ens = net_ens_all[:n_ens]

    def ens_net_make_prediction(x_list):
        preds = np.array([net.predict(x, verbose=2) for x,net in zip(x_list,net_ens)])
        return preds



    spreads_net_dists = []
    errors_net_ens_dists = []
    preds = np.array([modelrun_test[:n_test,:,np.newaxis] for _ in range(n_ens)])
    for i in range(0,int(200//lead_time)+1):

        spread = np.mean(np.std(preds,axis=0),axis=1) # std over ensemble, mean over gridpoints
        spreads_net_dists.append(spread)
        ensmean = np.mean(preds, axis=0)
        truth = modelrun_test[i*lead_time:i*lead_time+n_test]
        error_ens = np.sqrt(np.mean((np.squeeze(ensmean) - truth)**2,axis=1))
        errors_net_ens_dists.append(error_ens)
        preds = ens_net_make_prediction(preds)

    spreads_net_dists = np.array(spreads_net_dists).squeeze()
    errors_net_ens_dists = np.array(errors_net_ens_dists)

    for pert_scale in [0.001,0.003,0.01, 0.03,0.1,0.3,1, 3]:


        param_string = f'Nlorenz{N}_n_svs{n_svs}_n_ens{n_ens}_trainnoise{noise_on_traindata}_testnoise{noise_on_testdata}_' + \
                       f'svdleadtime{svd_leadtime}_noise{noise}_pertscale{pert_scale}_ntest{n_test}'

        x = modelrun_test[465]
        # plot one example forecast
        x_perts = create_pertubed_states(x)
        ens_fc = network.predict(x_perts[:,:,np.newaxis]).squeeze()
        plt.figure()
        plt.plot(ens_fc.T)
        preds = x_perts[:,:,np.newaxis]
        preds_ctrl = x[np.newaxis,:,np.newaxis]
        # add control member
        res = [preds.squeeze()]
        res_ctrl = [preds_ctrl.squeeze()]
        for i in range(1,int(200//lead_time)+1):
            preds = network.predict(preds)
            preds_ctrl = network.predict(preds_ctrl)
            res.append(preds.squeeze())
            res_ctrl.append(preds_ctrl.squeeze())

        res = np.array(res)
        res_ctrl = np.array(res_ctrl)
        # plot single gridpoint
        plt.figure()
        plt.plot(leadtimes, res_ctrl[:,0], label='ctr', linewidth=2)
        plt.plot(leadtimes, res[:,:,0], color='black')
        plt.legend()
        sns.despine()
        plt.xlabel('leadtime [MTU]')
        plt.savefig(f'example_forecast_{param_string}.svg')



        # it is much faster to predict all initial states in one go (because then only 1 tensorflow call is necessary
        # as opposed to looping over initial states in python

        # SVD ensemble

        pert_inits =  np.array([ create_pertubed_states(x) for x in tqdm(modelrun_test[:n_test])])
        # now has shape (n_test,n_ens,N)
        pert_inits = np.transpose(pert_inits, axes=(1,0,2))
        preds = pert_inits[:,:,:,np.newaxis]
        preds_ctrl = modelrun_test[:n_test,:,np.newaxis]
        spreads_dists = []
        errors_dists = []
        errors_ens_dists = []
        for i in range(0,int(200//lead_time)+1):

            spread = np.mean(np.std(preds,axis=0),axis=1) # std over ensemble, mean over gridpoints
            spreads_dists.append(spread.squeeze())
            ensmean = np.mean(preds, axis=0)
            truth = modelrun_test[i*lead_time:i*lead_time+n_test]
            error_ens = np.sqrt(np.mean((np.squeeze(ensmean) - truth)**2,axis=1))
            errors_ens_dists.append(error_ens)
            error_ctrl = np.sqrt(np.mean((np.squeeze(preds_ctrl) - truth)**2,axis=1))
            errors_dists.append(error_ctrl)

            preds_new = np.array([network.predict(preds[iens, :, :]) for iens in range(n_ens)])
            preds = preds_new
            preds_ctrl = network.predict(preds_ctrl)

        spreads_dists = np.array(spreads_dists)
        errors_dists = np.array(errors_dists)
        errors_ens_dists = np.array(errors_ens_dists)


        # random perturbations
        def random_perturbation(x):
            perts = np.random.normal(0, scale=pert_scale, size=(n_ens,N))
            # since this is a finite sample from the distribution,
            # it does not necessarily have 0 mean and pert_scale variance (section 4.5 in Tellus paper)
            # and we therefore rescale them
            perts = (perts - perts.mean(axis=0)) / perts.std(axis=0) * pert_scale
            # perts is (n_ens,N), and x is (N), so we can use standard numpy broadcasting
            # to create n_ens states
            x_pert = x + perts
            assert(np.allclose(np.mean(x_pert-x, axis=0),0.))
            assert(np.allclose(np.std(x_pert, axis=0),pert_scale))
            return x_pert

        pert_inits =  np.array([ random_perturbation(x) for x in tqdm(modelrun_test[:n_test])])
        pert_inits = np.transpose(pert_inits, axes=(1,0,2))
        preds = pert_inits[:,:,:,np.newaxis]
        spreads_rand_dists = []
        errors_rand_dists = []
        for i in range(0,int(200//lead_time)+1):

            spread = np.mean(np.std(preds,axis=0),axis=1) # std over ensemble, mean over gridpoints
            spreads_rand_dists.append(spread.squeeze())
            ensmean = np.mean(preds, axis=0)
            truth = modelrun_test[i*lead_time:i*lead_time+n_test]
            error_ens = np.sqrt(np.mean((np.squeeze(ensmean) - truth)**2,axis=1))
            errors_rand_dists.append(error_ens)

            preds_new = np.array([network.predict(preds[iens, :, :]) for iens in range(n_ens)])
            preds = preds_new


        spreads_rand_dists = np.array(spreads_rand_dists)
        errors_rand_dists = np.array(errors_rand_dists)



        corrs_svd= []
        corrs_rand = []
        corrs_netens= []
        for ltime in range(200//lead_time):
            corrs_svd.append(np.corrcoef(spreads_dists[ltime], errors_ens_dists[ltime])[0,1])
            corrs_rand.append(np.corrcoef(spreads_rand_dists[ltime], errors_rand_dists[ltime])[0,1])
            corrs_netens.append(np.corrcoef(spreads_net_dists[ltime], errors_net_ens_dists[ltime])[0, 1])


        plt.figure()
        plt.plot(leadtimes, errors_dists.mean(axis=1), label='rmse ctrl', color='black')
        plt.plot(leadtimes, errors_ens_dists.mean(axis=1), label='rmse ensmean svd', color='#1b9e77')
        plt.plot(leadtimes, spreads_dists.mean(axis=1), label='spread svd', linestyle='--', color='#1b9e77')
        plt.plot(leadtimes, errors_net_ens_dists.mean(axis=1), label='rmse ensmean netens', color='#d95f02')
        plt.plot(leadtimes, spreads_net_dists.mean(axis=1), label='spread netens', linestyle='--', color='#d95f02')
        plt.plot(leadtimes, errors_rand_dists.mean(axis=1), label='rmse randpert', color='#7570b3')
        plt.plot(leadtimes, spreads_rand_dists.mean(axis=1), label='spread randpert', linestyle='--', color='#7570b3')
        plt.legend()
        plt.xlabel('leadtime [MTU]')
        sns.despine()
        plt.title(f'n_ens:{n_ens}')
        plt.savefig(f'leadtime_vs_spread_error_netens{param_string}.svg')



        plt.figure()
        plt.plot(leadtimes[1:], corrs_svd, label='svd', color='#1b9e77')
        plt.plot(leadtimes[1:], corrs_netens, label='netens', color='#d95f02')
        plt.plot(leadtimes[1:], corrs_rand, label='randpert', color='#7570b3')
        plt.legend()
        plt.ylim((-0.2,1))
        plt.xlabel('leadtime [MTU]')
        plt.ylabel('correlation')
        sns.despine()
        plt.title(f'n_ens:{n_ens}')
        plt.savefig(f'leadtime_vs_error_spread_corr_netens_{param_string}.svg')


        pickle.dump((errors_dists, errors_ens_dists ,spreads_dists ,errors_net_ens_dists, spreads_net_dists),
                    open(f'{param_string}.pkl','wb'))

        plt.close('all')
        df = pd.DataFrame({'leadtime':leadtimes,
                           # we have rmse and std, so to average it, we have to square it, and than at
                           # the and take the squareroot again
                           'rmse_ctrl':np.sqrt(np.mean(errors_dists**2,1)).squeeze(),
                           'rmse_ensmean_svd':np.sqrt(np.mean(errors_ens_dists**2,1)).squeeze(),
                           'rmse_ensmean_rand':np.sqrt(np.mean(errors_rand_dists**2,1)).squeeze(),
                           'rmse_ensmean_netens':np.sqrt(np.mean(errors_net_ens_dists**2,1)).squeeze(),
                           'spread_svd':np.sqrt(np.mean(spreads_dists**2,1)).squeeze(),
                           'spread_rand':np.sqrt(np.mean(spreads_rand_dists**2,1)).squeeze(),
                           'spread_netens':np.sqrt(np.mean(spreads_net_dists**2,1)).squeeze(),
                           # the corrs arrays dont have an entry for leadtome=0, we we add 0 here to
                           # have the same length
                           'corr_svd':[0] + corrs_svd,
                           'corr_rand':[0] + corrs_rand,
                           'corr_netens':[0] + corrs_netens,
                           'n_ens':n_ens,
                           'pert_scale':pert_scale,
                           })
        df.to_csv(f'{outdir}/lorenz95_spread_error_correlation_{param_string}.csv')




