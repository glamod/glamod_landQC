#!/bin/bash
set -x

function write_and_submit_bastion_script {
    parallel_script=${1}

    # generate a "screen" instance in detached mode
    screen -S "plots" -d -m

    # run the parallel script in this detached screen
    screen -r "plots" -X stuff $'conda activate glamod_QC \n'

    # run the parallel script in this detached screen
    #  restrict to 1 job so that these are done in sequence
    screen -r "plots" -X stuff $"parallel --jobs 1 < ${parallel_script} "


} # write_and_submit_bastion_script

cwd=$(pwd)
SCRIPT_DIR=${cwd}/parallel_scripts/

parallel_script="${SCRIPT_DIR}/parallel_plots.bash"

if [ -e "${parallel_script}" ]; then
    rm "${parallel_script}"
fi

echo "python plot_map_of_flagging_rates.py" >> ${parallel_script}

write_and_submit_bastion_script "${parallel_script}"
