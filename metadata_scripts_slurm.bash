#!/bin/bash -l
#SBATCH -p ProdQ
#SBATCH -A glamod
#SBATCH -N 1
#SBATCH -o /ichec/home/users/rdunn/glamod_landQC/logs/metadata.out
#SBATCH -e /ichec/home/users/rdunn/glamod_landQC/logs/metadata.err 
#SBATCH -t 48:00:00
#SBATCH --mail-user=robert.dunn@metoffice.gov.uk
#SBATCH --mail-type=BEGIN,END

# activate python environment
module load conda
conda activate glamod_QC

python make_inventory.py
python plot_inventory.py
python make_station_listing.py
python plot_station_years.py
python plot_station_record_map.py
