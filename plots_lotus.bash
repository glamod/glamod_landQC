#!/bin/bash
#BSUB -q long-serial
#BSUB -J QC_plots
#BSUB -o /home/users/rjhd2/c3s311a_lot2/code/glamod_landQC/logs/plots.out
#BSUB -e /home/users/rjhd2/c3s311a_lot2/code/glamod_landQC/logs/plots.err 
#BSUB -W 40:00
#BSUB -R "rusage[mem=2000] select[type==any]"
#BSUB -M 2000

module load jaspy

python plot_map_of_flagging_rates.py
