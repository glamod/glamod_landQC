#!/bin/bash -l
#SBATCH -p ProdQ
#SBATCH -A glamod
#SBTACH -N 1
#SBATCH -o /ichec/home/users/${USER}/glamod_landQC/logs/plots.out
#SBATCH -e /ichec/home/users/${USER}/glamod_landQC/logs/plots.err 
#SBATCH -t 120:00

# activate python environment
module load conda
conda activate glamod_QC

# make the logs directory
if [ ! -d /ichec/home/users/${USER}/glamod_landQC/logs/ ]; then
    mkdir -p /ichec/home/users/${USER}/glamod_landQC/logs/
fi
    

python plot_map_of_flagging_rates.py
