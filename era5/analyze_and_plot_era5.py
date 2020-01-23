

import os
import pandas as pd
import seaborn as sns
import numpy as np
from pylab import plt

plotdir='plots/'
os.system(f'mkdir -p {plotdir}')


df = []
for n_ens in [2,4,10,20,100]:
    for pert_scale in [0.001,0.003,0.01, 0.03,0.1,0.3,1, 3]:
        ifile=f'output/era5_leadtime_vs_skill_and_spread_era5_2.5deg_weynetal_1979-2016_2017-2018_n_svs10_n_ens{n_ens}_pertscale{pert_scale}_nresample1_svdleadtime8.csv'

        _df = pd.read_csv(ifile, index_col=0)
        df.append(_df)

df = pd.concat(df, sort=False)


# find optimal pert_scale, indpendently for svd and rand ensemble, via picking
# the one where the improvement in ensemble mean rmse is best, for n_ens=10
leadtime=96

n_ens=20
sub = df.query(f'leadtime==@leadtime & n_ens==@n_ens')

best_pert_scale_rand = sub['pert_scale'].iloc[sub['mse_ensmean_rand'].values.argmin()]
best_pert_scale_svd = sub['pert_scale'].iloc[sub['mse_ensmean_svd'].values.argmin()]
# alternative: select on maximum spread-error-correlation
# best_pert_scale_rand = sub['pert_scale'].iloc[sub['corr_rand'].values.argmax()]
# best_pert_scale_svd = sub['pert_scale'].iloc[sub['corr_svd'].values.argmax()]

# note: the values for netens ar eindependent of selection, so wen can take them from any of the sub dataframes

sub_svd = df.query('pert_scale==@best_pert_scale_svd & n_ens==@n_ens')
sub_rand = df.query('pert_scale==@best_pert_scale_rand& n_ens==@n_ens')

sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.rcParams['savefig.bbox']='tight'
figsize=(7,4)
plt.figure(figsize=figsize)
# we use the first qualitative colorblind friendly palette from http://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=3)
plt.plot(sub_svd['leadtime'], sub_svd['rmse_ctrl'], label='rmse ctrl', color='black')
plt.plot(sub_svd['leadtime'], sub_svd['mse_ensmean_svd'], label='rmse ensmean svd', color='#1b9e77')
plt.plot(sub_rand['leadtime'], sub_rand['mse_ensmean_rand'], label='rmse ensmean rand', color='#7570b3')
plt.plot(sub_rand['leadtime'], sub_rand['mse_ensmean_netens'], label='rmse ensmean netens', color='#d95f02')
# plt.plot(suball['leadtime'], suball['mse_pers'], label='rmse persistence')
plt.plot(sub_svd['leadtime'], sub_svd['spread_svd'], label='spread svd', color='#1b9e77', linestyle='--')
plt.plot(sub_rand['leadtime'], sub_rand['spread_rand'], label='spread rand', color='#7570b3', linestyle='--')
plt.plot(sub_svd['leadtime'], sub_svd['spread_netens'], label='spread netens', color='#d95f02', linestyle='--')
plt.legend()
plt.xlabel('leadtime [h]')
sns.despine()
plt.title(f'n_ens:{n_ens}')
plt.savefig(f'{plotdir}/era5_leadtime_vs_skill_and_spread_best_n_ens{n_ens}.svg')

#plot correlation. we omit leadtime==0, because here correlation does not make sense
plt.figure(figsize=figsize)
plt.plot(sub_svd['leadtime'][1:], sub_svd['corr_svd'][1:], label='svd', color='#1b9e77')
plt.plot(sub_rand['leadtime'][1:], sub_rand['corr_rand'][1:], label='rand', color='#7570b3')
plt.plot(sub_svd['leadtime'][1:], sub_svd['corr_netens'][1:], label='netens', color='#d95f02')
plt.legend()
plt.xlabel('leadtime [h]')
plt.ylabel('spread-error correlation')
sns.despine()
plt.title(f'n_ens:{n_ens}')
plt.savefig(f'{plotdir}/era5_skill_spread_corr_best_n_ens{n_ens}.svg')



#%%
#$$ ensmean skill vs n_ens
# we use the selection based on n_ens=
leadtime=60
sub_svd = df.query('pert_scale==@best_pert_scale_svd & leadtime==@leadtime')
sub_rand = df.query('pert_scale==@best_pert_scale_rand & leadtime==@leadtime')
sub_netens = sub_rand.query('n_ens<=20') # netens we only have up to 20 members (the rest are dubplicates from 20 members)


plt.figure(figsize=figsize)
plt.plot(sub_svd['n_ens'], sub_svd['mse_ensmean_svd'], label='rmse ensmean svd', color='#1b9e77')
plt.plot(sub_rand['n_ens'], sub_rand['mse_ensmean_rand'], label='rmse ensmean rand', color='#7570b3')
plt.plot(sub_netens['n_ens'], sub_netens['mse_ensmean_netens'], label='rmse ensmean netens', color='#d95f02')
plt.plot(sub_svd['n_ens'], sub_svd['spread_svd'], label='spread svd', color='#1b9e77', linestyle='--')
plt.plot(sub_rand['n_ens'], sub_rand['spread_rand'], label='spread rand', color='#7570b3', linestyle='--')
plt.plot(sub_netens['n_ens'], sub_netens['spread_netens'], label='spread netens', color='#d95f02', linestyle='--')
plt.legend()
plt.xlabel('N ensemble members')
plt.title(f'leadtime={leadtime} h')
sns.despine()
plt.savefig(f'{plotdir}/era5_n_ens_vs_skill_leadtime{leadtime}.svg')


plt.figure(figsize=figsize)
plt.plot(sub_svd['n_ens'], sub_svd['corr_svd'], label='ensmean svd', color='#1b9e77')
plt.plot(sub_rand['n_ens'], sub_rand['corr_rand'], label='ensmean rand', color='#7570b3')
plt.plot(sub_netens['n_ens'], sub_netens['corr_netens'], label='ensmean netens', color='#d95f02')
plt.legend()
plt.xlabel('N ensemble members')
plt.ylabel('spread error correlation')
plt.title(f'leadtime={leadtime} h')
sns.despine()
plt.savefig(f'{plotdir}/era5_n_ens_vs_corr_leadtime{leadtime}.svg')
