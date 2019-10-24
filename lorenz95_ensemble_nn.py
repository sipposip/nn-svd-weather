"""

needs tensorflow >=2.0

https://www.sciencedirect.com/science/article/pii/S0012825212000657
https://www.tandfonline.com/doi/abs/10.1111/j.1600-0870.2006.00197.x
https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.1994.0105
https://rmets.onlinelibrary.wiley.com/doi/epdf/10.1002/qj.49712252905
http://www.elsevierscitech.com/emails/physics/climate/Ensemble_forecasting.pdf

"""

import matplotlib
# matplotlib.use('agg')

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
# # set maximum numpber of CPUs to use
# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16,
#                        allow_soft_placement=True)
# session = tf.compat.v1.Session(config=config)



# paramters for experiments
N = 40  # number of variables
F = 8
# Nsteps = 10000
Nsteps = 10000
Nspinup = 500
Nsteps = Nsteps + Nspinup
tstep=0.01
t_arr = np.arange(0, Nsteps) * tstep
lead_time = 10
# fixed params neural network
n_epoch = 30
noise = 0.1

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
modelrun_train = modelrun_train + np.random.normal(scale=noise, size=modelrun_train.shape)
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

preds=modelrun_test
#  we have to add an empty channel dimension
preds = preds[..., np.newaxis]
errors = []
accs = []
tsteps = []
net_eval = []
error_dists = []
for i in range(1,int(200//lead_time)+1):
    print(i)
    preds = network.predict(preds)
    truth = modelrun_test[i*lead_time:]
    preds_cut = np.squeeze(preds[:-i*lead_time])
    assert(preds_cut.shape==truth.shape)
    rmse = np.sqrt(np.mean( (preds_cut-truth)**2))
    errors.append(rmse)
    tsteps.append(i*lead_time)
    rmse_per_fc =  np.sqrt(np.mean( (preds_cut-truth)**2, axis=1))
    error_dists.append(rmse_per_fc)
    acc = np.mean([np.corrcoef(truth[i],preds_cut[i])[0,1] for i in range(len(preds_cut))])
    accs.append(acc)
    net_eval.append(pd.DataFrame({'lead_time_training':lead_time,'lead_time':tsteps,'rmse':errors,
                             'acc':accs}))


# plot distributions of errors
plt.figure()
sns.distplot(error_dists[0], label='1')
sns.distplot(error_dists[9], label='10')
plt.legend()
plt.ylabel('rmse')
sns.despine()
plt.savefig('error_dists_L95.svg')

## gradients
svd_leadtime= 4 #https://www.tandfonline.com/doi/pdf/10.1111/j.1600-0870.2006.00197.x?needAccess=true use 0.4 timeunits
                # here we define it in multiples of the network timestep (which is 10 times the model timestep of 0.01
                #, so one means 0.1
n_svs = 10
pert_scale = noise * 2 / norm_std
n_ens = 10

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

x = X_train[0]


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


x_perts = create_pertubed_states(x)

plt.figure()
plt.plot(x_perts.T)

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
plt.plot(res_ctrl[:,0], label='ctr', linewidth=2)
plt.plot(res[:,:,0], color='black')
plt.legend()
plt.savefig('example_forecast.svg')

spreads_dists = []
errors_dists = []
errors_ens_dists = []
n_test = 500
for i_init in tqdm(range(n_test)):
    x = modelrun_test[i_init]
    x_perts = create_pertubed_states(x, leadtime=svd_leadtime)
    preds = np.expand_dims(x_perts, axis=-1)
    pred_ctrl = np.expand_dims(x,axis=0)
    pred_ctrl = np.expand_dims(pred_ctrl,axis=-1)
    spreads_per_leadtime = []
    errors_per_leadtime = []
    errors_ens_per_leadtime = []
    for i in range(1,int(200//lead_time)+1):
        preds = network.predict(preds)
        pred_ctrl = network.predict(pred_ctrl)
        spread = np.mean(np.std(preds,axis=0))
        spreads_per_leadtime.append(spread)
        ensmean = np.mean(preds, axis=0)
        truth = modelrun_test[i_init+i*lead_time]
        error = np.sqrt(np.mean((np.squeeze(pred_ctrl) - truth)**2))
        errors_per_leadtime.append(error)
        error_ens = np.sqrt(np.mean((np.squeeze(ensmean) - truth)**2))
        errors_ens_per_leadtime.append(error_ens)
    spreads_dists.append(spreads_per_leadtime)
    errors_dists.append(errors_per_leadtime)
    errors_ens_dists.append(errors_ens_per_leadtime)

spreads_dists = np.array(spreads_dists).T
errors_dists = np.array(errors_dists).T
errors_ens_dists = np.array(errors_ens_dists).T

plt.figure()
plt.plot(errors_dists.mean(axis=1), label='rmse ctrl')
plt.plot(errors_ens_dists.mean(axis=1), label='rmse ensmean')
plt.plot(spreads_dists.mean(axis=1), label='spread')
plt.legend()
plt.savefig('leadtime_vs_spread_error.svg')

corrs= []
for ltime in range(200//lead_time):
    plt.ioff()
    plt.figure()
    plt.scatter(spreads_dists[ltime], error_dists[ltime][:n_test])
    plt.title(f'leadtime {(ltime+1)*lead_time}')
    plt.savefig(f'spread_vs_error_leadtime{(ltime+1)*lead_time}')
    plt.close()
    corrs.append(np.corrcoef(spreads_dists[ltime], error_dists[ltime][:n_test])[0,1])


plt.figure()
plt.plot(corrs)
plt.savefig('leadtime_vs_error_spread_corr.svg')

plt.figure()
sns.kdeplot(spreads_dists[0], label='1')
sns.kdeplot(spreads_dists[9], label='10')
plt.legend()
plt.savefig('kdeplot_spread')