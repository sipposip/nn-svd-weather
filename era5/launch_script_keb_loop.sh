

for mem in 1 2 3 4 5 6 7 8 9 10; do
sbatch <<EOF
#! /bin/bash

#SBATCH -A SNIC2018-3-545
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:1

/pfs/nobackup/home/s/sebsc/miniconda3/bin/python train_era5_2.5deg_weynetal_batch.py ${mem}
EOF
done