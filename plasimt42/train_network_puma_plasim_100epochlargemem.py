
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:25:01 2018

train a CNN for puma. The cnn configuration used is the one obtained
in the tuning process.



input: number of training years to be used (passed in via argv), and index (in years)
at which the training data shall start (with respect to the full training set)
e.g. if train_year=10, and train_offset = 30, then years 30-40 from the training set will be used

@author: sebastian
"""

import os
import sys
import pickle
import json

import numpy as np
import pandas as pd
import xarray as xr

from keras.layers import Convolution2D,Dropout
from keras import layers
import keras
import tensorflow as tf


from dask.diagnostics import ProgressBar
ProgressBar().register()

modelname=sys.argv[1]
train_years = int(sys.argv[2])
i_train = int(sys.argv[3])
load_data_lazily = sys.argv[4] == "True"
train_years_offset = 0
# load_data_lazily: if False, the trainign data is loaded completely int RAM, if True, we use 
# only lazily loaded dask arrays that are loaded while training.
# if the data is too big to fit into ram, this has to be set to True

# the input file was produced with the preprocessing script.
# it is the full puma output, put into one file, reordered coordinates to
# ('time','lat','lon','lev') and all variables stacked along the 'lev' dimension


os.system('mkdir -p data')

ifile='/proj/bolinc/users/x_sebsc/gcm_complexity_machinelearning/models/preprocessed/'+modelname+'reordered.merged.nc'
#ifile='/climstorage/sebastian/gcm_complexity_machinelearning/modelruns/old/plasim_t21/plasimt21.reordered.normalized.merged.nc'

N_gpu = 0   # 0 if you want to train on CPUs (will be automatically parallelized to all available CPUs)

lead_time = 1 # days, lead time used for training



# if False, the trainign data is loaded completely int RAM, if True, we use 
# only lazily loaded dask arrays that are loaded while training.
# if the data is too big to fit into ram, this has to be set to True



test_years=30  # we actually dont need the test data here, but we need the amount
                # of test data in order to not include in the training

days_per_year_per_model={'pumat21':360,'pumat31':360,'pumat42':360,'pumat21_noseas':360,
                         'plasimt21':365,'plasimt31':365,'plasimt42':365,
                         'pumat42_regridt21':360,
                         'plasimt42_regridt21':365} # noteL we are ignoring leap years in plasim here,
                                                                            
days_per_year = days_per_year_per_model[modelname]

N_test = days_per_year * test_years
N_train = days_per_year * train_years
N_train_offset = days_per_year * train_years_offset


## parameters for the neural network

# fixed (not-tuned params)
batch_size = 32
num_epochs = 100
pool_size = 2
drop_prob=0
conv_activation='relu'

## the params came out of the tuning process (for pumat21, Scher 2018)
params = {'conv_depth': 32, 'hidden_size': 500,
          'kernel_size': 6, 'lr': 0.0001, 'n_hidden_layers': 0}


# create a parameter string that will be used for naming of files
param_string = modelname +'_'+ '_'.join([str(e) for e in (train_years,train_years_offset,num_epochs,lead_time, i_train)])


def prepare_data(x,lead_time):
    ''' split up data in predictor and predictant set by shifting
     it according to the given lead time, and then split up
     into train, developement and test set'''
    if lead_time == 0:
        X = x
        y = X[:]
    else:

        X = x[:-lead_time]
        y = x[lead_time:]

    X_train = X[N_test+train_years_offset:N_test+train_years_offset+N_train]
    y_train = y[N_test+train_years_offset:N_test+train_years_offset+N_train]

    X_test = X[:N_test]
    y_test = y[:N_test]

    return X_train,y_train, X_test, y_test


print('open inputdata')

data = xr.open_dataarray(ifile, chunks={'time':1})  # we have to define chunks,
# then the data is opened as dask -array (out of core)

# note that the time-variable in the input file is confusing: it contains the
# day of the year of the simulation, thus it is repeating all the time
# (it loops from 1 to 360 and then jumps back to one)





# check that we have enough data for the specifications
if N_train + N_test > data.shape[0]:
    raise Exception('not enough timesteps in input file!')


Nlat,Nlon,n_channels=data.shape[1:4]


X_train,y_train, X_test, y_test = prepare_data(data,lead_time)


# normalization: we normalize all data so that the training set has zero
# mean and unit variance on all levels. here we have  a problem with float32 data
# https://github.com/pydata/xarray/issues/1346, so we cast it to float 64
print('compute normalization weights on training data')
norm_mean = X_train.astype('float64').mean(('time','lat','lon'))
norm_std = X_train.astype('float64').std(('time','lat','lon'))

# no we have to cast back to float32, otherwise the normalization will cast the whole data to float64
norm_mean = norm_mean.astype('float32')
norm_std = norm_std.astype('float32')

# due to limitation in dask/xarray, it is faster to compute the normalization
# mean and std no, save it to disk and load it again (see https://github.com/dask/dask/issues/874)

norm_mean.to_netcdf('data/norm_mean_'+param_string+'.nc')
norm_std.to_netcdf('data/norm_std_'+param_string+'.nc')

norm_mean = xr.open_dataarray('data/norm_mean_'+param_string+'.nc')
norm_std = xr.open_dataarray('data/norm_std_'+param_string+'.nc')

if not load_data_lazily:
    print('load train data into memory')
    X_train.load()
    y_train.load()





# now normalize everything
X_train = (X_train - norm_mean) / norm_std
y_train = (y_train - norm_mean) / norm_std
# we actually dont need the test data in this script, but for clarity we write out
# everything here. since X_test and y_test is lazily loaded, this does not take 
# take significant computation time anyway
X_test = (X_test - norm_mean) / norm_std
y_test = (y_test - norm_mean) / norm_std


# to speed up the training with lazily loaded data, we copy the training data to the local SSD of the computing
# node (/scratch/local/)

X_train.to_netcdf('/scratch/local/X_train.nc')
y_train.to_netcdf('/scratch/local/y_train.nc')

X_train = xr.open_dataarray('/scratch/local/X_train.nc',chunks={'time':1})
y_train = xr.open_dataarray('/scratch/local/y_train.nc',chunks={'time':1})




def build_model(conv_depth, kernel_size, hidden_size, n_hidden_layers, lr):

    model = keras.Sequential([

        ## Convolution with dimensionality reduction (similar to Encoder in an autoencoder)
        Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation, input_shape=(Nlat,Nlon,n_channels)),
        layers.MaxPooling2D(pool_size=pool_size),
        Dropout(drop_prob),
        Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
        layers.MaxPooling2D(pool_size=pool_size),
        # end "encoder"
    
    
        # dense layers (flattening and reshaping happens automatically)
        ] + [layers.Dense(hidden_size, activation='sigmoid') for i in range(n_hidden_layers)] +
    
        [
    
    
        # start "Decoder" (mirror of the encoder above)
        Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
        layers.UpSampling2D(size=pool_size),
        Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
        layers.UpSampling2D(size=pool_size),
        layers.Convolution2D(n_channels, kernel_size, padding='same', activation=None)
        ]
        )


    
    optimizer= keras.optimizers.adam(lr=lr)

    if N_gpu > 1:
        with tf.device("/cpu:0"):
            # convert the model to a model that can be trained with N_GPU GPUs
             model = keras.utils.multi_gpu_model(model, gpus=N_gpu)

    model.compile(loss='mean_squared_error', optimizer = optimizer)

    
    return model




model = build_model(**params)

print(model.summary())

best_weight_file='data/bestweights_'+param_string+'leadtime'+str(lead_time)+'_trainyeras'+str(train_years)+'.h5'


print('start training')
hist = model.fit(X_train, y_train,
                   batch_size = batch_size,
         verbose=1,
         epochs = num_epochs,
         validation_split=0.1,
         callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                    min_delta=0,
                                    patience=5, # just to make sure we use a lot of patience before stopping
                                    verbose=0, mode='auto'),
                   keras.callbacks.ModelCheckpoint(best_weight_file, monitor='val_loss',
                                                verbose=1, save_best_only=True,
                                                save_weights_only=True, mode='auto', period=1)]
         )

print('finished training')

# get best model from the training (based on validation loss),
# this is neccessary because the early stopping callback saves the model "patience" epochs after the best one

model.load_weights(best_weight_file)

# remove the file created by ModelCheckppoint
os.system('rm '+best_weight_file)


# save the model weights and architecture 
model.save_weights('data/weights_'+param_string+'.h5')
json.dump(model.to_json(), open('data/modellayout_'+param_string+'.json','w'))


# reformat history

hist =  hist.history

pickle.dump(hist,open('data/train_history_params_'+param_string+'.pkl','wb'))

hist_df = pd.DataFrame(hist)
hist_df.to_csv('data/train_history_params_'+param_string+'.csv')
