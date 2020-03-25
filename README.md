# nn-svd-weather
This repository contains the code for the paper "Ensemble neural network forecasts with singular value decomposition" by Sebastian Scher and Gabriele Messori.
For Era 5 and for Plasimt42, there are separate scripts for
1) training the neural networks
2) computing the jacobians and the singular vectors
3) doing and evaluating the forecasts
4) make plots
for Lorenz95, 3) and 4) are combined into a single script.
for Lorenz95,1 and 2) as well as 3) and 4) are combined into single scripts.

The repository also contains the output from step 3) (in the folders "output" and "data") and all plots.

###Lorenz96
training, jacobians and singular vectors: ```lorenz95/lorenz95_ensemble_nn_train.py```
forecasts and plots: ```lorenz95/lorenz95_analyze_and_plot.py```
###Plasim
training: ```plasimt42/train_network_puma_plasim_100epochlargemem.py```
jacobians and singular vectors: ```plasimt42/plasim_precompute_jacobians.py```
forecast and evaluation: ```plasimt42/plasim_from_precomputed_jacobians_keb.py```
plots: ```plasimt42/analyze_and_plot_plasimt42.py```
###Era5
data download: ```era5/download_era5_z500.py```
precomputation of normalization weights: ```era5/era5_compute_normalization_weigths.py```
training: ```era5/train_era5_2.5deg_weynetal_batch.py```
selecting best members from training: ```era5/era5_select_best_members.py```
jacobians and singular vectors: ```era5/svd_ensemble_era5_2.5deg_weynetal_precompute_jacobians.py```
forecast and evaluation: ```era5/svd_ensemble_era5_2.5deg_weynetal_from_precomputed_jacobians.py```
plots: ```era5/analyze_and_plot_era5.py```


Each subfolder contains a *.yml file that contains the anaconda environment used for each model.
