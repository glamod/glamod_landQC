#!/bin/bash
set -x

function write_and_submit_bastion_script {
    parallel_script=${1}

    # generate a "screen" instance in detached mode
    screen -S "metadata" -d -m

    # run the parallel script in this detached screen
    screen -r "metadata" -X stuff $'conda activate glamod_QC \n'

    # run the parallel script in this detached screen
    #  restrict to 1 job so that these are done in sequence
    screen -r "metadata" -X stuff $"parallel --jobs 1 < ${parallel_script} "


} # write_and_submit_bastion_script

parallel_script="${SCRIPT_DIR}/parallel_metadata.bash"

echo "conda activate glamod_QC" > ${parallel_script}
echo "" > ${parallel_script}
echo "python make_inventory.py" > ${parallel_script}
echo "python plot_inventory.py" > ${parallel_script}
echo "python make_station_listing.py" > ${parallel_script}
echo "python plot_station_years.py" > ${parallel_script}
echo "python plot_station_record_map.py" > ${parallel_script}

write_and_submit_bastion_script "${parallel_script}"
