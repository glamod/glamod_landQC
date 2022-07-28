#!/bin/bash -l
#SBATCH --partition=short-serial-4hr
#SBATCH --account=short4hr
#SBATCH --job-name=QC_plots
#SBATCH --output=/home/users/rjhd2/c3s311a_lot2/code/glamod_landQC/logs/plots.out
#SBATCH --error=/home/users/rjhd2/c3s311a_lot2/code/glamod_landQC/logs/plots.err 
#SBATCH --time=120:00
#SBATCH --mem=2000

# Note - only use jaspy for this.
#   Get segmentation faults on save when using venv - not sure why
module load jaspy

python plot_map_of_flagging_rates.py
