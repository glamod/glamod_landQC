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

python make_inventory.py
python plot_inventory.py
python make_station_listing.py
python plot_station_years.py
python plot_station_record_map.py
