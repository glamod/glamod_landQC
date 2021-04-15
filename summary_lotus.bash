#!/bin/bash -l
#SBATCH --qos=short-serial
#SBATCH --job-name=QC_plots
#SBATCH --output=/home/users/rjhd2/c3s311a_lot2/code/glamod_landQC/logs/summary.out
#SBATCH --error=/home/users/rjhd2/c3s311a_lot2/code/glamod_landQC/logs/summary.err 
#SBATCH --time=360:00
#SBATCH --mem=2000

module load jaspy

python summary_counts_per_year.py --stage I
