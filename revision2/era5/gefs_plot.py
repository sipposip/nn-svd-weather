

import os
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import xarray as xr
from pylab import plt

# regrid=True
regrid=False
if regrid:
    regridstr = "_regridTrue"
else:
    regridstr = ""

data = []
for n_ens in range(2,11):
    ifile = f'gefs/gefs_reforecast_scores_n_ens{n_ens}{regridstr}.pkl'
    res = pickle.load(open(ifile,'rb'))
    # the data is on float32. convert all float data to float 64
    for key in res.keys():
        if type(res[key]) == np.ndarray or type(res[key]) == xr.DataArray and res[key].dtype=='float32':
            res[key] = res[key].astype('float64')
    data.append(res)




df = []
for sub in data:
    for ltime in range(len(sub['leadtime'])):

        sub['spread'][ltime] = sub['spread'][ltime].astype('float64')
        if ltime > 0:  # for leadtime 0, correlation does not make sense
            # for gefs "spread" is std, not var, so we sqrt only mse
            corr =np.corrcoef(np.sqrt(sub['mse_ensmean'][ltime].squeeze()),
                                            sub['spread'][ltime].squeeze())[0,1]
    
        else:
            corr = [0]
    
        _df = pd.DataFrame({'leadtime':sub['leadtime'][ltime],
                            'corr':corr,
                            #compute RMSE and mean stddev
                            'rmse_ensmean': np.sqrt(np.mean(sub['mse_ensmean'][ltime])).values,
                            'rmse_ctrl': np.sqrt(np.mean(sub['mse_ctrl'][ltime])).values,
                            'crps': np.mean(sub['crps'][ltime]),
                            'spread': np.sqrt(np.mean(sub['spread'][ltime]**2)).values,
                            'n_ens':sub['n_ens']
                            }, index=[0]
                           )
        df.append(_df)
df = pd.concat(df, sort=False)

df['leadtime'] = df['leadtime'].astype('int')

sns.set_context("paper", font_scale=1.5, rc={'lines.linewidth': 2.5,
                                             'legend.fontsize':'small'})
plt.rcParams['savefig.bbox']='tight'
plt.rcParams['legend.frameon']=True
plt.rcParams['legend.framealpha']=0.4
plt.rcParams['legend.labelspacing']=0.1
figsize=(7.5,3)

sub = df.query('n_ens==10')
plt.figure(figsize=(20,11))
plt.subplot(3, 2, 1)
# convert rmse to m2/a2
plt.plot(sub['leadtime'], sub['rmse_ensmean']*9.81, label='ensmean', color='black')
plt.plot(sub['leadtime'], sub['rmse_ctrl']*9.81, label='ctrl')
plt.plot(sub['leadtime'], sub['spread']*9.81, label='spread',color='black', linestyle='--')
plt.legend(loc='upper left', fontsize=14)
plt.ylabel('rmse (solid) \n spread (dashed) [$m^2/s^2$]')
plt.xlabel('leadtime [h]')
plt.xlabel('leadtime [h]')
sns.despine()
plt.suptitle(f'GEFS reforecasts')

plt.subplot(3, 2, 5)
plt.plot(sub['leadtime'], sub['crps']*9.81, color='black')
plt.ylabel('crps [$m^2/s^2$]')
plt.xlabel('leadtime [h]')
plt.xlabel('leadtime [h]')
sns.despine()

plt.subplot(3, 2, 3)
plt.plot(sub['leadtime'][1:], sub['corr'][1:], label='svd', color='black')

plt.xlabel('leadtime [h]')
plt.ylabel('spread-error correlation')
sns.despine()
plt.ylim((0, 0.9))
plt.xlim((0,120))


## n_ens vs rmse / corr/ crps
sub_df = df
# here we omit leadtime 0 (because we dont need and it makes the plots confusing, especially for corr)
sub_df = sub_df[sub_df['leadtime']!=0]
plt.subplot(3,2,2)
sns.lineplot('n_ens','rmse_ensmean', data=sub_df, hue='leadtime', legend='full')
sns.lineplot('n_ens','spread', data=sub_df, hue='leadtime', style=True, dashes=[(2,2)], legend=False)
plt.legend(bbox_to_anchor=(0, 1.1), loc=2, borderaxespad=0., ncol=6)
plt.xlabel('')
plt.ylabel('rmse (solid) \n spread (dashed) [$m^2/s^2$]')
sns.despine()
plt.subplot(3,2,4)
sns.lineplot('n_ens','corr', data=sub_df, hue='leadtime', legend='full')
plt.legend(bbox_to_anchor=(0, 1.1), loc=2, borderaxespad=0., ncol=6)
plt.xlabel('')
plt.ylabel('spread error correlation')
sns.despine()
plt.subplot(3,2,6)
sns.lineplot('n_ens','crps', data=sub_df, hue='leadtime', legend='full')
plt.legend(bbox_to_anchor=(0, 1.1), loc=2, borderaxespad=0., ncol=6)
plt.xlabel('$n_{ens}$')
plt.ylabel('crps [$m^2/s^2$]')
sns.despine()
plt.tight_layout()
plt.savefig(f'plots/gefs_skill_regrid{regrid}.pdf')
plt.savefig(f'plots/gefs_skill_regrid{regrid}.svg')
plt.savefig(f'plots/gefs_skill_regrid{regrid}.png')
