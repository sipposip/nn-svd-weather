#!/bin bash

for n_ens in 20 100; do
    for pert_scale in 0.001 0.01 0.1 1; do
        sbatch plasim_from_precomputed_jacobians_keb_batch.py ${n_ens} ${pert_scale}
    done
done