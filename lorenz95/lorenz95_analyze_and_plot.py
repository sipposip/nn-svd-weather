

import os
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from pylab import plt

plotdir='plots/'
os.system(f'mkdir -p {plotdir}')




# paramters for experiments
N = 200  # number of variables
F = 8
noise = 0.1
n_test = 500
noise_on_traindata = True
noise_on_testdata = True
svd_leadtime= 4
n_svs = 10


full_res = []
for n_ens in [2,4,10,20,100]:
    for pert_scale in [0.001,0.003,0.01, 0.03,0.1,0.3,1, 3]:
        param_string = f'Nlorenz{N}_n_svs{n_svs}_n_ens{n_ens}_trainnoise{noise_on_traindata}_testnoise{noise_on_testdata}_' + \
                       f'svdleadtime{svd_leadtime}_noise{noise}_pertscale{pert_scale}_ntest{n_test}'
        ifile = f'data/lorenz95_spread_error_correlation_{param_string}.pkl'
        res = pickle.load(open(ifile,'rb'))
        # the data is on float32. convert all float data to float 64
        for key in res.keys():
            if type(res[key]) == np.ndarray and res[key].dtype=='float32':
                res[key] = res[key].astype('float64')
        full_res.append(res)


def bootstrapped_correlation_difference(x1,y1,x2,y2, perc=5, N=1000):
    """ compute the difference between corr(x1,y1) and corr (x2,y2) plus
    bands [perc,100-perc] of the differecne in  correlation, estimated via bootstrpaping.
    This method is useful when the uncertainty between the two correlations is not independent.
    x1,y1,x2,y2 must all have the same length"""
    assert (len(x1) == len(y1))
    assert (len(x1) == len(y2))
    assert (len(x2) == len(x1))
    n_samples = len(x1)
    diff_corrs = []
    # we use the same indices both for sample 1 and sample 2, and then compute the difference in correlation
    for i in range(N):
        indices = np.random.choice(n_samples, replace=True, size=n_samples)
        corr1 = np.corrcoef(x1[indices], y1[indices])[0, 1]
        corr2 = np.corrcoef(x2[indices], y2[indices])[0, 1]
        diff_corrs.append(corr1-corr2)
    corrs = np.array(diff_corrs)
    meancorrdiff = np.corrcoef(x1, y1)[0, 1] - np.corrcoef(x2, y2)[0, 1]
    upper = np.percentile(corrs, q=100 - perc)
    lower = np.percentile(corrs, q=perc)
    assert(upper>=lower)
    return meancorrdiff, lower, upper


def bootstrapped_rmse_difference(x1,x2, perc=5, N=1000):
    """ compute difference between x1 and x2 plus uncertainty, when x1 and x2 are either rmse or standarddeviation

    """
    assert(len(x1)==len(x2))
    n_samples = len(x1)
    means = []
    for i in range(N):
        indices = np.random.choice(n_samples, replace=True, size=n_samples)
        # now compute difference in  RMSE on this subsample
        mm = np.sqrt(np.mean(x1[indices]**2)) - np.sqrt(np.mean(x2[indices]**2))
        means.append(mm)
    means = np.array(means)
    mmean = np.sqrt(np.mean(x1**2)) - np.sqrt(np.mean(x2**2))
    upper = np.percentile(means, q=100 - perc)
    lower = np.percentile(means, q=perc)
    # assert (upper >= lower) # we deactivate this check here because if one or both of x1 and x2
    # concist only of repreated values, then numerical inaccuracis can lead to
    # lower being a tiny little larger than upper (even though they should be the same in this case)
    return np.array([mmean, lower, upper])



def corr(x1,x2):
    return np.corrcoef(x1,x2)[0,1]

# loop over n_ens and pert_scale
res_df = []
for sub in full_res:
    # for each leadtime, compute mean error, spread, and spread error correlaation, the latter
    # including uncertainty estimates
    for ltime in range(len(sub['leadtime'])):

        if ltime > 0:  # for leadtime 0, correlation does not make sense
            # we want the correlation nbetween rmse and spread, so we have to take the sqrt since we have
            # mse and variance
            corr_svd = corr(np.sqrt(sub['mse_ensmean_svd'][ltime].squeeze()),
                                                        np.sqrt(sub['spread_svd'][ltime].squeeze()))
            corr_netens = corr(np.sqrt(sub['mse_ensmean_netens'][ltime].squeeze()),
                                                        np.sqrt(sub['spread_netens'][ltime].squeeze()))
            corr_rand = corr(np.sqrt(sub['mse_ensmean_rand'][ltime].squeeze()),
                                                        np.sqrt(sub['spread_rand'][ltime].squeeze()))

        else:
            corr_svd, corr_netens, corr_rand = 0,0,0
            corrdiff_svd_rand, corrdiff_svd_netens, corrdiff_rand_netens = [0,0,0],[0,0,0],[0,0,0]

        _df = pd.DataFrame({'leadtime':sub['leadtime'][ltime],
                            'corr_svd':corr_svd,
                            'corr_rand':corr_rand,
                            'corr_netens': corr_netens,

                            #compute RMSE and mean stddev
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


# find optimal pert_scale, indpendently for svd and rand ensemble, via picking
# the one where the improvement in ensemble mean rmse is best, for n_ens=10
leadtime=1 # MTUs

n_ens=20
sub = df.query(f'leadtime==@leadtime & n_ens==@n_ens')

best_pert_scale_rand = sub['pert_scale'].iloc[sub['rmse_ensmean_rand'].values.argmin()]
best_pert_scale_svd = sub['pert_scale'].iloc[sub['rmse_ensmean_svd'].values.argmin()]
# alternative: select on maximum spread-error-correlation
# best_pert_scale_rand = sub['pert_scale'].iloc[sub['corr_rand'].values.argmax()]
# best_pert_scale_svd = sub['pert_scale'].iloc[sub['corr_svd'].values.argmax()]

# note: the values for netens ar eindependent of selection, so wen can take them from any of the sub dataframes

sub_svd = df.query('pert_scale==@best_pert_scale_svd')
sub_rand = df.query('pert_scale==@best_pert_scale_rand')
# make a new df, containing the results of the best pert_scale for each method
# netens and ctrl are independent of this, so we can take them from either of the
# two dfs
df_best = sub_svd.copy()
df_best['rmse_ensmean_rand'] = sub_rand['rmse_ensmean_rand']
df_best['corr_rand'] = sub_rand['corr_rand']
df_best['spread_rand'] = sub_rand['spread_rand']
# pert_scale is now mixed, so it is meaningles and we can delete it
df_best.drop(columns=['pert_scale'], inplace=True)

# now compute uncertainty intervals for differences
# for this, we need the "raw" data again (not the mean errors, but the errors per forecast,
# now for the best pert_scale per method
# again, for netens, it does not matter, since it is the same anywary
# for n_ens in df_best['n_ens'].unique():
#     for ltime in range(len(df_best['leadtime'].unique())):
all_leadtimes_orig_order = np.array(full_res[0]['leadtime'])
df_new = []
for i,row in df_best.iterrows():
    _n_ens = row['n_ens']
    _leadtime = row['leadtime']
    ltime = int(np.where(all_leadtimes_orig_order==_leadtime)[0])

    raw_data_svd = full_res[int(np.where([e['pert_scale'] == best_pert_scale_svd and e['n_ens']==_n_ens for e in full_res])[0])]
    raw_data_rand = full_res[int(np.where([e['pert_scale'] == best_pert_scale_rand and e['n_ens']==_n_ens for e in full_res])[0])]
    # select data for this leadtime only
    rmse_svd = np.sqrt(raw_data_svd['mse_ensmean_svd'])[ltime].squeeze()
    rmse_rand = np.sqrt(raw_data_rand['mse_ensmean_rand'])[ltime].squeeze()
    rmse_netens = np.sqrt(raw_data_svd['mse_ensmean_netens'])[ltime].squeeze()
    spread_svd = np.sqrt(raw_data_svd['spread_svd'])[ltime].squeeze()
    spread_rand = np.sqrt(raw_data_rand['spread_rand'])[ltime].squeeze()
    spread_netens = np.sqrt(raw_data_svd['spread_netens'])[ltime].squeeze()
    # check whether the netens data is really the same for both subsets
    assert(np.array_equal(raw_data_svd['mse_ensmean_netens'], raw_data_rand['mse_ensmean_netens']))

    # compute differences in forecast error plus uncertainty.
    # to do this proberly, we have to use MSE, and variance instead
    errordiff_svd_rand = bootstrapped_rmse_difference(rmse_svd, rmse_rand)
    errordiff_svd_netens = bootstrapped_rmse_difference(rmse_svd, rmse_netens)
    errordiff_rand_netens = bootstrapped_rmse_difference(rmse_rand, rmse_netens)

    # compute differences in forecast spread plus uncertainty
    spreaddiff_svd_rand = bootstrapped_rmse_difference(spread_svd, spread_rand)
    spreaddiff_svd_netens = bootstrapped_rmse_difference(spread_svd, spread_netens)
    spreaddiff_rand_netens = bootstrapped_rmse_difference(spread_rand, spread_netens)

    # compute differences in correlation plus uncertainty
    if ltime > 0:
        corrdiff_svd_rand = bootstrapped_correlation_difference(rmse_svd, spread_svd, rmse_rand, spread_rand)
        corrdiff_svd_netens = bootstrapped_correlation_difference(rmse_svd, spread_svd, rmse_netens, spread_netens)
        corrdiff_rand_netens = bootstrapped_correlation_difference(rmse_rand, spread_rand, rmse_netens, spread_netens)
    else:
        corrdiff_svd_rand, corrdiff_svd_netens, corrdiff_rand_netens = [0, 0, 0], [0, 0, 0], [0, 0, 0]

    # now update this row with the new variables, and add it to the list of tows for the new dataframe
    row['errordiff_svd_rand'] = errordiff_svd_rand[0]
    row['errordiff_svd_rand_lower'] = errordiff_svd_rand[1]
    row['errordiff_svd_rand_upper'] = errordiff_svd_rand[2]
    row['errordiff_svd_netens'] = errordiff_svd_netens[0]
    row['errordiff_svd_netens_lower'] = errordiff_svd_netens[1]
    row['errordiff_svd_netens_upper'] = errordiff_svd_netens[2]
    row['errordiff_rand_netens'] = errordiff_rand_netens[0]
    row['errordiff_rand_netens_lower'] = errordiff_rand_netens[1]
    row['errordiff_rand_netens_upper'] = errordiff_rand_netens[2]

    row['spreaddiff_svd_rand'] = spreaddiff_svd_rand[0]
    row['spreaddiff_svd_rand_lower'] = spreaddiff_svd_rand[1]
    row['spreaddiff_svd_rand_upper'] = spreaddiff_svd_rand[2]
    row['spreaddiff_svd_netens'] = spreaddiff_svd_netens[0]
    row['spreaddiff_svd_netens_lower'] = spreaddiff_svd_netens[1]
    row['spreaddiff_svd_netens_upper'] = spreaddiff_svd_netens[2]
    row['spreaddiff_rand_netens'] = spreaddiff_rand_netens[0]
    row['spreaddiff_rand_netens_lower'] = spreaddiff_rand_netens[1]
    row['spreaddiff_rand_netens_upper'] = spreaddiff_rand_netens[2]

    row['corrdiff_svd_rand'] = corrdiff_svd_rand[0]
    row['corrdiff_svd_rand_lower'] = corrdiff_svd_rand[1]
    row['corrdiff_svd_rand_upper'] = corrdiff_svd_rand[2]
    row['corrdiff_svd_netens'] = corrdiff_svd_netens[0]
    row['corrdiff_svd_netens_lower'] = corrdiff_svd_netens[1]
    row['corrdiff_svd_netens_upper'] = corrdiff_svd_netens[2]
    row['corrdiff_rand_netens'] = corrdiff_rand_netens[0]
    row['corrdiff_rand_netens_lower'] = corrdiff_rand_netens[1]
    row['corrdiff_rand_netens_upper'] = corrdiff_rand_netens[2]
    df_new.append(pd.DataFrame(row))
# this is now a list of rows. therefore, we need to concat it along axis1, and then
# transpose the whole dataframe with .T
df_new = pd.concat(df_new, sort=False, axis=1).T
df_best = df_new

# now we computed the differences in an extra step. as sanity check, compare to the differences of the means
#, where the means were computed in the first loop.
np.testing.assert_allclose(df_best['rmse_ensmean_svd'] - df_best['rmse_ensmean_rand'], df_best['errordiff_svd_rand'])
np.testing.assert_allclose(df_best['rmse_ensmean_svd'] - df_best['rmse_ensmean_netens'], df_best['errordiff_svd_netens'])
np.testing.assert_allclose(df_best['rmse_ensmean_rand'] - df_best['rmse_ensmean_netens'], df_best['errordiff_rand_netens'])
np.testing.assert_allclose(df_best['spread_svd'] - df_best['spread_rand'], df_best['spreaddiff_svd_rand'])
np.testing.assert_allclose(df_best['spread_svd'] - df_best['spread_netens'], df_best['spreaddiff_svd_netens'])
np.testing.assert_allclose(df_best['spread_rand'] - df_best['spread_netens'], df_best['spreaddiff_rand_netens'])
np.testing.assert_allclose(df_best['corr_svd'] - df_best['corr_rand'], df_best['corrdiff_svd_rand'])
np.testing.assert_allclose(df_best['corr_svd'] - df_best['corr_netens'], df_best['corrdiff_svd_netens'])
np.testing.assert_allclose(df_best['corr_rand'] - df_best['corr_netens'], df_best['corrdiff_rand_netens'])


## plotting
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.rcParams['savefig.bbox']='tight'
plt.rcParams['legend.frameon']=True
plt.rcParams['legend.framealpha']=0.4

sub_df = df_best.query('n_ens==@n_ens')
figsize=(7.5,3)
# we use the first qualitative colorblind friendly palette from
# http://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=3)
#

# mean forecast error and spread vs leadtime, for fixed n_ens
plt.figure(figsize=figsize)
plt.plot(sub_df['leadtime'], sub_df['rmse_ctrl'], label='rmse ctrl', color='black')
plt.plot(sub_df['leadtime'], sub_df['rmse_ensmean_svd'], label='rmse ensmean svd', color='#1b9e77')
plt.plot(sub_df['leadtime'], sub_df['rmse_ensmean_rand'], label='rmse ensmean rand', color='#7570b3')
plt.plot(sub_df['leadtime'], sub_df['rmse_ensmean_netens'], label='rmse ensmean netens', color='#d95f02')
plt.plot(sub_df['leadtime'], sub_df['spread_svd'], label='spread svd', color='#1b9e77', linestyle='--')
plt.plot(sub_df['leadtime'], sub_df['spread_rand'], label='spread rand', color='#7570b3', linestyle='--')
plt.plot(sub_df['leadtime'], sub_df['spread_netens'], label='spread netens', color='#d95f02', linestyle='--')
plt.legend()
plt.xlabel('leadtime [MTU]')
sns.despine()
plt.title(f'n_ens:{n_ens}')
plt.savefig(f'{plotdir}/lorenz95_leadtime_vs_skill_and_spread_best_n_ens{n_ens}.svg')


# difference mean forecast error and spread vs leadtime plus uncertainty, for fixed n_ens
plt.figure(figsize=figsize)
plt.plot(sub_df['leadtime'], sub_df['errordiff_svd_rand'], label='rmse svd-rand', color='#1b9e77')
plt.fill_between(sub_df['leadtime'], sub_df['errordiff_svd_rand_lower'],sub_df['errordiff_svd_rand_upper'],
                 color='#1b9e77', alpha=0.5)
plt.plot(sub_df['leadtime'], sub_df['errordiff_svd_netens'], label='rmse svd-netens', color='#7570b3')
plt.fill_between(sub_df['leadtime'], sub_df['errordiff_svd_netens_lower'],sub_df['errordiff_svd_netens_upper'],
                 color='#7570b3', alpha=0.5)
plt.plot(sub_df['leadtime'], sub_df['errordiff_rand_netens'], label='rmse rand-netens', color='#d95f02')
plt.fill_between(sub_df['leadtime'], sub_df['errordiff_rand_netens_lower'],sub_df['errordiff_rand_netens_upper'],
                 color='#d95f02', alpha=0.5)
plt.plot(sub_df['leadtime'], sub_df['spreaddiff_svd_rand'], label='spread svd-rand', color='#1b9e77', linestyle='--')
plt.fill_between(sub_df['leadtime'], sub_df['spreaddiff_svd_rand_lower'],sub_df['spreaddiff_svd_rand_upper'],
                 color='#1b9e77', alpha=0.5, linestyle='--')
plt.plot(sub_df['leadtime'], sub_df['spreaddiff_svd_netens'], label='spread svd-netens', color='#7570b3', linestyle='--')
plt.fill_between(sub_df['leadtime'], sub_df['spreaddiff_svd_netens_lower'],sub_df['spreaddiff_svd_netens_upper'],
                 color='#7570b3', alpha=0.5, linestyle='--')
plt.plot(sub_df['leadtime'], sub_df['spreaddiff_rand_netens'], label='spread rand-netens', color='#d95f02', linestyle='--')
plt.fill_between(sub_df['leadtime'], sub_df['spreaddiff_rand_netens_lower'],sub_df['spreaddiff_rand_netens_upper'],
                 color='#d95f02', alpha=0.5, linestyle='--')
plt.legend()
plt.xlabel('leadtime [MTU]')
plt.axhline(0, color='black', zorder=-5, linewidth=1)
sns.despine()
plt.title(f'n_ens:{n_ens}')
plt.savefig(f'{plotdir}/lorenz95_leadtime_vs_skill_and_spread_diff_best_n_ens{n_ens}.svg')



# spread error correlation vs leadtime
# plot correlation. we omit leadtime==0, because here correlation does not make sense
plt.figure(figsize=figsize)
plt.plot(sub_df['leadtime'][1:], sub_df['corr_svd'][1:], label='svd', color='#1b9e77')
plt.plot(sub_df['leadtime'][1:], sub_df['corr_rand'][1:], label='rand', color='#7570b3')
plt.plot(sub_df['leadtime'][1:], sub_df['corr_netens'][1:], label='netens', color='#d95f02')
plt.legend()
plt.xlabel('leadtime [MTU]')
plt.ylabel('spread-error correlation')
sns.despine()
plt.title(f'n_ens:{n_ens}')
plt.savefig(f'{plotdir}/lorenz95_skill_spread_corr_best_n_ens{n_ens}.svg')

# difference in spread error correlation vs leadtime
plt.figure(figsize=figsize)
plt.plot(sub_df['leadtime'][1:], sub_df['corrdiff_svd_rand'][1:], label='svd-rand', color='#1b9e77')
plt.fill_between(sub_df['leadtime'][1:], sub_df['corrdiff_svd_rand_lower'][1:],sub_df['corrdiff_svd_rand_upper'][1:],
                 color='#1b9e77', alpha=0.5)
plt.plot(sub_df['leadtime'][1:], sub_df['corrdiff_svd_netens'][1:], label='svd-netens', color='#7570b3')
plt.fill_between(sub_df['leadtime'][1:], sub_df['corrdiff_svd_netens_lower'][1:],sub_df['corrdiff_svd_netens_upper'][1:],
                 color='#7570b3', alpha=0.5)
plt.plot(sub_df['leadtime'][1:], sub_df['corrdiff_rand_netens'][1:], label='rand-netens', color='#d95f02')
plt.fill_between(sub_df['leadtime'][1:], sub_df['corrdiff_rand_netens_lower'][1:],sub_df['corrdiff_rand_netens_upper'][1:],
                 color='#d95f02', alpha=0.5)
plt.legend()
plt.xlabel('leadtime [MTU]')
plt.ylabel('spread-error correlation')
plt.axhline(0, color='black', zorder=-5, linewidth=1)
sns.despine()
plt.title(f'n_ens:{n_ens}')
plt.savefig(f'{plotdir}/lorenz95_skill_spread_diffcorr_best_n_ens{n_ens}.svg')



#%%
#
# we use the selection based on n_ens=
leadtime=1
sub_df = df_best.query('leadtime==@leadtime')
sub_netens = sub_df.query('n_ens<=20') # netens we only have up to 20 members (the rest are dubplicates from 20 members)

# ensmean skill vs n_ens
plt.figure(figsize=figsize)
plt.plot(sub_df['n_ens'], sub_df['rmse_ensmean_svd'], label='mse ensmean svd', color='#1b9e77')
plt.plot(sub_df['n_ens'], sub_df['rmse_ensmean_rand'], label='mse ensmean rand', color='#7570b3')
plt.plot(sub_netens['n_ens'], sub_netens['rmse_ensmean_netens'], label='mse ensmean netens', color='#d95f02')
plt.plot(sub_df['n_ens'], sub_df['spread_svd'], label='spread svd', color='#1b9e77', linestyle='--')
plt.plot(sub_df['n_ens'], sub_df['spread_rand'], label='spread rand', color='#7570b3', linestyle='--')
plt.plot(sub_netens['n_ens'], sub_netens['spread_netens'], label='spread netens', color='#d95f02', linestyle='--')
plt.legend()
plt.xlabel('N ensemble members')
plt.title(f'leadtime={leadtime} MTU')
sns.despine()
plt.savefig(f'{plotdir}/lorenz95_n_ens_vs_skill_leadtime{leadtime}.svg')

# difference ensmean skill vs n_ens
plt.figure(figsize=figsize)
plt.plot(sub_df['n_ens'], sub_df['errordiff_svd_rand'], label='svd-rand', color='#1b9e77')
plt.fill_between(sub_df['n_ens'], sub_df['errordiff_svd_rand_lower'], sub_df['errordiff_svd_rand_upper'],
                 alpha=0.5,color='#1b9e77')
plt.plot(sub_netens['n_ens'], sub_netens['errordiff_svd_netens'], label='svd-netens', color='#7570b3')
plt.fill_between(sub_netens['n_ens'], sub_netens['errordiff_svd_netens_lower'], sub_netens['errordiff_svd_netens_upper'],
                alpha=0.5,color='#7570b3')
plt.plot(sub_netens['n_ens'], sub_netens['errordiff_rand_netens'], label='rand-netens', color='#d95f02')
plt.fill_between(sub_netens['n_ens'], sub_netens['errordiff_rand_netens_lower'], sub_netens['errordiff_rand_netens_upper'],
                alpha=0.5,color='#d95f02')
plt.legend()
plt.xlabel('N ensemble members')
plt.title(f'leadtime={leadtime} MTU')
plt.axhline(0, color='black', zorder=-5, linewidth=1)
sns.despine()
plt.savefig(f'{plotdir}/lorenz95_n_ens_vs_skill_diff_leadtime{leadtime}.svg')


## spread-error correlation vs n_ens
plt.figure(figsize=figsize)
plt.plot(sub_df['n_ens'], sub_df['corr_svd'], label='ensmean svd', color='#1b9e77')
plt.plot(sub_df['n_ens'], sub_df['corr_rand'], label='ensmean rand', color='#7570b3')
plt.plot(sub_netens['n_ens'], sub_netens['corr_netens'], label='ensmean netens', color='#d95f02')
plt.legend()
plt.xlabel('N ensemble members')
plt.ylabel('spread error correlation')
plt.title(f'leadtime={leadtime} MTU')
sns.despine()
plt.savefig(f'{plotdir}/lorenz95_n_ens_vs_corr_leadtime{leadtime}.svg')

## difference i n spread-error correlation vs n_ens
plt.figure(figsize=figsize)
plt.plot(sub_df['n_ens'], sub_df['corrdiff_svd_rand'], label='svd-rand', color='#1b9e77')
plt.fill_between(sub_df['n_ens'], sub_df['corrdiff_svd_rand_lower'], sub_df['corrdiff_svd_rand_upper'],
                 alpha=0.5,color='#1b9e77')
plt.plot(sub_netens['n_ens'], sub_netens['corrdiff_svd_netens'], label='svd-netens', color='#7570b3')
plt.fill_between(sub_netens['n_ens'], sub_netens['corrdiff_svd_netens_lower'], sub_netens['corrdiff_svd_netens_upper'],
                alpha=0.5,color='#7570b3')
plt.plot(sub_netens['n_ens'], sub_netens['corrdiff_rand_netens'], label='rand-netens', color='#d95f02')
plt.fill_between(sub_netens['n_ens'], sub_netens['corrdiff_rand_netens_lower'], sub_netens['corrdiff_rand_netens_upper'],
                alpha=0.5,color='#d95f02')
plt.legend()
plt.xlabel('N ensemble members')
plt.ylabel('diff spread error correlation')
plt.title(f'leadtime={leadtime} MTU')
plt.axhline(0, color='black', zorder=-5, linewidth=1)
sns.despine()
plt.savefig(f'{plotdir}/lorenz95_n_ens_vs_corrdiff_leadtime{leadtime}.svg')
