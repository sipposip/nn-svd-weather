#! /climstorage/sebastian/anaconda3/envs/xesmf-env/bin/ipython

import numpy as np
from pylab import plt
import xarray as xr
from dask.diagnostics import ProgressBar
import xesmf as xe
import pandas as pd
import os
if not os.path.exists('plots/'):
    os.mkdir('plots/')

plt.rcParams['savefig.bbox']='tight'


norm_weights_folder = 'normalization_weights/'
# in the make and eval script, we forgot to scale CRPS to real units (it was computed on the
# normalized data). therefore we need the norm_scale here
norm_std = xr.open_dataarray(norm_weights_folder+'/normalization_weights_era5_2.5deg_2015-2016_tres6_geopotential_500_std.nc').values
norm_mean = xr.open_dataarray(norm_weights_folder+'/normalization_weights_era5_2.5deg_2015-2016_tres6_geopotential_500_mean.nc').values

pert_scale_rand = 0.1
pert_scale_svd = 0.3
p_drop=0.001
svd_leadtime = int(24/6)
n_svs_reduced = 60

n_ens=100
leadtime=72
outdir='/climstorage/sebastian/nn_ensemble_nwp/output/'
ifile_rand =f'{outdir}/predictions_rand_ensmean_era5_2.5deg_weynetal_1979-2016_2017-2018_n_ens{n_ens}_pertscale{pert_scale_rand}_nresample1_leadtime{leadtime}.npz'
ifile_netens =f'{outdir}/predictions_netens_ensmean_era5_2.5deg_weynetal_1979-2016_2017-2018_nresample1_n_ens20_leadtime{leadtime}.npz'
ifile_svd =f'{outdir}/predictions_svd_ensmean_era5_2.5deg_weynetal_1979-2016_2017-2018_n_svs100_n_ens{n_ens}_pertscale{pert_scale_svd}_nresample1_svdleadtime{svd_leadtime}_n_svs_reduced{n_svs_reduced}_leadtime{leadtime}.npz'


data_rand = np.load(ifile_rand)['arr_0'].squeeze()
data_netens = np.load(ifile_netens)['arr_0'].squeeze()
data_svd = np.load(ifile_svd)['arr_0'].squeeze()

data_rand = data_rand * norm_std + norm_mean
data_netens = data_netens * norm_std + norm_mean
data_svd = data_svd * norm_std + norm_mean

init_dates = pd.date_range('20170101-00:00','20181231-18:00', freq=f'{6*8}h')
# remove last couple of dates, since we dont have them
init_dates = init_dates[:-10]

# load gefs reforecatss

ProgressBar().register()
ifile = '/climstorage/sebastian/nn_ensemble_nwp/gefs_reforecast/hgt_pres_latlon_all_20170101_20181231_NH.nc'
data = xr.open_dataset(ifile, chunks={'time':100})['Geopotential_height']

data = data.sel(time=slice('20170101', '20181231'))
# remove empty pressure dimension
data = data.squeeze()

data = data.sel(fhour=pd.to_timedelta(f'{leadtime}h'))
# ensemble mean
data = data.mean('ens')
data.load()

# convert to same units as era5
data = data*9.81

#regrid
target_grid = xr.open_dataset('/climstorage/sebastian/nn_ensemble_nwp/gefs_reforecast/constants_2.5deg.nc')
# only NH
target_grid = target_grid.isel(lat=target_grid['lat']>0)
regridder = xe.Regridder(data, target_grid.coords, 'bilinear')
data = regridder(data)


for i in range(0,len(init_dates),20):
    init_date = init_dates[i]
    gefs = data.sel(time=init_date)
    svd = data_svd[i]
    rand = data_rand[i]
    netens = data_netens[i]

    svd = xr.DataArray(data=svd, coords=gefs.coords, dims=gefs.dims)
    rand = xr.DataArray(data=rand, coords=gefs.coords, dims=gefs.dims)
    netens = xr.DataArray(data=netens, coords=gefs.coords, dims=gefs.dims)

    plt.figure()
    plt.subplot(221)
    gefs.plot()
    plt.title('gefs')
    plt.subplot(222)
    svd.plot()
    plt.title('svd')
    plt.subplot(223)
    rand.plot()
    plt.title('rand')
    plt.subplot(224)
    netens.plot()
    plt.title('multitrain')
    plt.suptitle(init_date.strftime('%Y%m%d'))

    plt.tight_layout()

    plt.savefig(f'plots/ensmean_snapshots_{i:03d}.png', dpi=300)
    plt.savefig(f'plots/ensmean_snapshots_{i:03d}.pdf')
