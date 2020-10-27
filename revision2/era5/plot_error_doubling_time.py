

import os
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from pylab import plt
from tqdm import tqdm
import xarray as xr

plotdir='plots/'
os.system(f'mkdir -p {plotdir}')

def savefig(figname):
    plt.savefig(figname+'.svg')
    plt.savefig(figname+'.pdf')

data_netens = []
# use only the data for all (=20) ensemble members
n_ens=20
ifile = f'output/era5_fc_eval_results_netens_era5_2.5deg_weynetal_1979-2016_2017-2018_nresample1_n_ens{n_ens}.pkl'
res = pickle.load(open(ifile,'rb'))
# the data is on float32. convert all float data to float 64
for key in res.keys():
    if type(res[key]) == np.ndarray and res[key].dtype=='float32':
        res[key] = res[key].astype('float64')
data = res

df = []
for ltime in range(len(data['leadtime'])):
    for imem in range(n_ens):
        _df = pd.DataFrame({'leadtime':data['leadtime'][ltime],
                            'rmse':np.sqrt(np.mean(data['mse_ensmean_netens_permember'][ltime,imem])),
                            'n_ens': data['n_ens'],
                            'member':imem
                            }, index=[0]
                           )
        df.append(_df)
df = pd.concat(df, sort=False)

da = df.set_index(['leadtime','member']).to_xarray()
dEdt = da.diff('leadtime', label='lower')
# remove leadtime zero (because our forecasts start from
#the analysis, we have zero RMSE at leadtime zero, even though
# this is of course not true since the analysis also is not perfect.
dEdt = dEdt.isel(leadtime=slice(1,None))
dEdt_divE = dEdt/da

#
# ## plotting
sns.set_context("paper", font_scale=1.5, rc={'lines.linewidth': 2.5,
                                             'legend.fontsize':'small'})
plt.rcParams['savefig.bbox']='tight'
plt.rcParams['legend.frameon']=True
plt.rcParams['legend.framealpha']=0.4
plt.rcParams['legend.labelspacing']=0.1
figsize=(7.5,3)



# -------------------- netens plots

## n_ens vs rmse / corr/ crps
plt.figure(figsize=(18,5))
plt.subplot(121)
for imem in dEdt['member']:
    plt.plot(dEdt['leadtime'],dEdt.isel(member=imem)['rmse'],
             color='black', alpha=0.6)
sns.despine()
plt.xlabel('leadtime [h]')
plt.ylabel('dE/dt [$m^2s^{-2}h^{-1}$]')
plt.subplot(122)
for imem in dEdt['member']:
    plt.plot(dEdt_divE['leadtime'],dEdt_divE.isel(member=imem)['rmse'],
             color='black', alpha=0.6)
sns.despine()
plt.xlabel('leadtime [h]')
plt.ylabel('(dE/dt)/E [$h^{-1}$]')
sns.despine()
savefig(f'{plotdir}/era5_netens_error_doubling_time')

plt.figure(figsize=(9,5))
plt.subplot(121)
sns.lineplot('leadtime','rmse', data=df, hue='member')
plt.legend()
sns.despine()
savefig(f'{plotdir}/era5_netens_rmse_permember')




