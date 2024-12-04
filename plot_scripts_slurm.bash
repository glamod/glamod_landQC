#!/bin/bash -l
#SBATCH -p ProdQ
#SBATCH -A glamod
#SBATCH -N 1
#SBATCH -o /ichec/home/users/rdunn/glamod_landQC/logs/plots.out
#SBATCH -e /ichec/home/users/rdunn/glamod_landQC/logs/plots.err 
#SBATCH -t 02:00:00
#SBATCH --mail-user=robert.dunn@metoffice.gov.uk
#SBATCH --mail-type=BEGIN,END

# activate python environment
module load conda
conda activate glamod_QC

# make the logs directory
if [ ! -d /ichec/home/users/rdunn/glamod_landQC/logs/ ]; then
    mkdir -p /ichec/home/users/rdunn/glamod_landQC/logs/
fi
    

python plot_map_of_flagging_rates.py
