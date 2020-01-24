#TODO: at the moment, the uncertainty for the correlatios is estimated independently for each
# method. this is not correct, becasue the uncertainties are not independent
# solution: do the bootstrapping the following way:
# -create bootstrapped sample of timeindices
# -compute the spread-error correlation for all methods on this bootstrap sample
# -compute the difference in correlation between the 3 methods, save this difference
# -compute the quantiles of these differences (over all bootstrap samples)
# the check the significanc on these (e.g. if for method1-method2 the 5-95 interval is completely positive, then
# method 1 has significantly higher correlation, if completely negative, then method 2 significantly higher, otherwise
# no significant difference

import os
from tqdm import tqdm
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from pylab import plt

plotdir='plots/'
os.system(f'mkdir -p {plotdir}')


#df = []
full_res = []
# for n_ens in [2,4,10,20,100]:
for n_ens in [2,4,10,20]: #TODO: include 100 !
    for pert_scale in [0.001,0.003,0.01, 0.03,0.1,0.3,1, 3]:
        # ifile=f'output/era5_leadtime_vs_skill_and_spread_era5_2.5deg_weynetal_1979-2016_2017-2018_n_svs10_n_ens{n_ens}_pertscale{pert_scale}_nresample1_svdleadtime8.csv'
        #
        # _df = pd.read_csv(ifile, index_col=0)
        # df.append(_df)

        ifile = f'output/era5_leadtime_vs_skill_and_spread_era5_2.5deg_weynetal_1979-2016_2017-2018_n_svs10_n_ens{n_ens}_pertscale{pert_scale}_nresample1_svdleadtime8.pkl'
        res = pickle.load(open(ifile,'rb'))
        full_res.append(res)

#df = pd.concat(df, sort=False)




def bootstrapped_correlation(x, y, perc=5, N=1000): #HACK!! TODO: set N higher!
    assert (len(x) == len(y))
    corrs = []
    for i in range(N):
        indices = np.random.choice(len(x), replace=True, size=len(x))
        corr = np.corrcoef(x[indices], y[indices])[0, 1]
        corrs.append(corr)
    corrs = np.array(corrs)
    # corrs = sns.algorithms.bootstrap(x,y,func=lambda x,y:np.corrcoef(x,y)[0,1])
    meancorr = np.corrcoef(x, y)[0, 1]
    upper = np.percentile(corrs, q=100 - perc)
    lower = np.percentile(corrs, q=perc)

    return meancorr, lower, upper

# loop over n_ens and pert_scale
res_df = []
for sub in tqdm(full_res):
    # for each leadtime, compute mean error, spread, and spread error correlaation, the latter
    # including uncertainty estimates
    for ltime in range(len(sub['leadtime'])):
        corr_svd,lower_svd,upper_svd = bootstrapped_correlation(np.sqrt(sub['mse_ensmean_svd'][ltime].squeeze()),
                                                    np.sqrt(sub['spread_svd'][ltime].squeeze()))

        corr_netens,lower_netens,upper_netens = bootstrapped_correlation(np.sqrt(sub['mse_ensmean_netens'][ltime].squeeze()),
                                                    np.sqrt(sub['spread_rand'][ltime].squeeze()))

        corr_rand,lower_rand,upper_rand = bootstrapped_correlation(np.sqrt(sub['mse_ensmean_rand'][ltime].squeeze()),
                                                    np.sqrt(sub['spread_rand'][ltime].squeeze()))

        _df = pd.DataFrame({'leadtime':sub['leadtime'][ltime],
                            'corr_svd':corr_svd,
                            'corr_svd_lower':lower_svd,
                            'corr_svd_upper':upper_svd,
                            'corr_rand':corr_rand,
                            'corr_rand_lower':lower_rand,
                            'corr_rand_upper':upper_rand,
                            'corr_netens': corr_netens,
                            'corr_netens_lower':lower_netens,
                            'corr_netens_upper':upper_netens,
                            #compute RAMSE and mean stddev
                            'rmse_ensmean_svd': np.sqrt(np.mean(sub['mse_ensmean_svd'][ltime])),
                            'rmse_ensmean_rand': np.sqrt(np.mean(sub['mse_ensmean_rand'][ltime])),
                            'rmse_ensmean_netens': np.sqrt(np.mean(sub['mse_ensmean_netens'][ltime])),
                            'spread_svd': np.sqrt(np.mean(sub['spread_svd'][ltime])),
                            'spread_rand': np.sqrt(np.mean(sub['spread_rand'][ltime])),
                            'spread_netens': np.sqrt(np.mean(sub['spread_netens'][ltime])),
                            'rmse_ctrl': np.sqrt(np.mean(sub['rmse_ctrl'][ltime])),
                            'n_ens': sub['n_ens'],
                            'pert_scale':sub['pert_scale'],
                            }, index=[0]
                           )
        res_df.append(_df)
df = pd.concat(res_df, sort=False)

# at leadtime 0, correlation dont make sense, therefore we set them all to zero
#TODO

# find optimal pert_scale, indpendently for svd and rand ensemble, via picking
# the one where the improvement in ensemble mean rmse is best, for n_ens=10
leadtime=96

n_ens=20
sub = df.query(f'leadtime==@leadtime & n_ens==@n_ens')

best_pert_scale_rand = sub['pert_scale'].iloc[sub['rmse_ensmean_rand'].values.argmin()]
best_pert_scale_svd = sub['pert_scale'].iloc[sub['rmse_ensmean_svd'].values.argmin()]
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
plt.plot(sub_svd['leadtime'], sub_svd['rmse_ensmean_svd'], label='rmse ensmean svd', color='#1b9e77')
plt.plot(sub_rand['leadtime'], sub_rand['rmse_ensmean_rand'], label='rmse ensmean rand', color='#7570b3')
plt.plot(sub_rand['leadtime'], sub_rand['rmse_ensmean_netens'], label='rmse ensmean netens', color='#d95f02')
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
plt.fill_between(sub_svd['leadtime'][1:], sub_svd['corr_svd_lower'][1:],sub_svd['corr_svd_upper'][1:],
                 alpha=0.5,color='#1b9e77')
plt.plot(sub_rand['leadtime'][1:], sub_rand['corr_rand'][1:], label='rand', color='#7570b3')
plt.fill_between(sub_rand['leadtime'][1:], sub_rand['corr_rand_lower'][1:],sub_rand['corr_rand_upper'][1:],
                 alpha=0.5, color='#7570b3')
# the netens is the same for all pert_scale combis, so we can for example pick it from sub_svd
plt.plot(sub_svd['leadtime'][1:], sub_svd['corr_netens'][1:], label='netens', color='#d95f02')
plt.fill_between(sub_svd['leadtime'][1:], sub_svd['corr_netens_lower'][1:],sub_svd['corr_netens_upper'][1:],
                 alpha=0.5,  color='#d95f02')
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
plt.plot(sub_svd['n_ens'], sub_svd['rmse_ensmean_svd'], label='rmse ensmean svd', color='#1b9e77')
plt.plot(sub_rand['n_ens'], sub_rand['rmse_ensmean_rand'], label='rmse ensmean rand', color='#7570b3')
plt.plot(sub_netens['n_ens'], sub_netens['rmse_ensmean_netens'], label='rmse ensmean netens', color='#d95f02')
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
plt.fill_between(sub_svd['n_ens'], sub_svd['corr_svd_lower'],sub_svd['corr_svd_upper'],
                 alpha=0.5,  color='#1b9e77')
plt.plot(sub_rand['n_ens'], sub_rand['corr_rand'], label='ensmean rand', color='#7570b3')
plt.fill_between(sub_rand['n_ens'], sub_rand['corr_rand_lower'],sub_rand['corr_rand_upper'],
                 alpha=0.5,  color='#7570b3')
plt.plot(sub_netens['n_ens'], sub_netens['corr_netens'], label='ensmean netens', color='#d95f02')
plt.fill_between(sub_netens['n_ens'], sub_netens['corr_netens_lower'],sub_netens['corr_netens_upper'],
                 alpha=0.5,  color='#d95f02')
plt.legend()
plt.xlabel('N ensemble members')
plt.ylabel('spread error correlation')
plt.title(f'leadtime={leadtime} h')
sns.despine()
plt.savefig(f'{plotdir}/era5_n_ens_vs_corr_leadtime{leadtime}.svg')
