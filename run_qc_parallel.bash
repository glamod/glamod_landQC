#!/bin/bash
#set -x
#******************************************************************
# Script to process all the stations.  Runs through station list
#   and submits each as a separate jobs on Bastion
#
# CALL
#    bash run_qc.bash STAGE WAIT CLOBBER
#
#    STAGE = I [internal] or N [neighbour]
#     WAIT = T [true] or F [false] # wait for upstream files to be ready
#  CLOBBER = C [clobber] or S [skip] # overwrite or skip existing files
#******************************************************************

#**************************************
# manage the input arguments
STAGE=$1
if [ "${STAGE}" != "I" ] && [ "${STAGE}" != "N" ]; then
    echo "Please enter valid switch. I (internal) or N (neighbour)"
    exit
fi
WAIT=$2
if [ "${WAIT}" != "T" ] && [ "${WAIT}" != "F" ]; then
    echo "Please enter valid waiting option. T (true - wait for upstream files) or F (false - skip missing files)"
    exit
fi
CLOBBER=$3
if [ "${CLOBBER}" != "C" ] && [ "${CLOBBER}" != "S" ]; then
    echo "Please enter valid clobber option. C (clobber - overwrite existing outputs) or S (skip - keep existing outputs)"
    exit
fi
# remove all 3 positional characters
shift
shift
shift

#**************************************
# other settings
cwd=$(pwd)
STATIONS_PER_BATCH=15000
N_JOBS=10

SCRIPT_DIR=${cwd}/parallel_scripts/
if [ ! -d "${SCRIPT_DIR}" ]; then
    mkdir "${SCRIPT_DIR}"
fi


#**************************************
# Set functions
function write_and_submit_bastion_script {
    parallel_script=${1}
    batch=${2}

    # generate a "screen" instance in detached mode
    screen -S "qc_${batch}" -d -m

    # run the parallel script in this detached screen
    screen -r "qc_${batch}" -X stuff $'conda activate glamod_QC \n'

    # run the parallel script in this detached screen
    screen -r "qc_${batch}" -X stuff $"parallel --jobs ${N_JOBS} < ${parallel_script}
"


} # write_and_submit_bastion_script

function prepare_parallel_script {
    batch=${1}

    if [ "${STAGE}" == "I" ]; then
	parallel_script="${SCRIPT_DIR}/parallel_internal_${batch}.bash"
    elif  [ "${STAGE}" == "N" ]; then
	parallel_script="${SCRIPT_DIR}/parallel_external_${batch}.bash"
    fi
    if [ -e "${parallel_script}" ]; then
	rm "${parallel_script}"
    fi
    echo "${parallel_script}"
} # prepare_parallel_script


#**************************************
# use configuration file to pull out paths &c
CONFIG_FILE="${cwd}/configuration.txt"

# VENVDIR="$(grep "venvdir " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
# using spaces after setting ID to ensure pull out correct line
# these are fixed references
ROOTDIR="$(grep "root " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"

# extract remaining locations
MFF_DIR="$(grep "mff " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
MFF_VER="$(grep "mff_version " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
MFF_ZIP="$(grep "in_compression " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
PROC_DIR="$(grep "proc " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
QFF_DIR="$(grep "qff " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
QFF_ZIP="$(grep "out_compression " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
VERSION="$(grep "version " "${CONFIG_FILE}" | awk -F'= ' 'FNR == 2 {print $2}')"
ERR_DIR="$(grep "errors " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
LOG_DIR="$(grep "logs " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
CONFIG_DIR="$(grep "config " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
IN_SUFFIX="$(grep "in_suffix " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
OUT_SUFFIX="$(grep "out_suffix " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
if [ ! -d "${ROOTDIR}${LOG_DIR}" ]; then
    mkdir "${ROOTDIR}${LOG_DIR}"
fi

#**************************************
# if neighbour checks make sure all files in place
if [ "${STAGE}" == "N" ]; then
    echo "${ROOTDIR}${CONFIG_DIR}${VERSION}neighbours.txt"
    if [ ! -f "${ROOTDIR}${CONFIG_DIR}${VERSION}neighbours.txt" ]; then
        read -p "Neighbour file missing - do you want to create? (Y/N): " run_neighbours

    else
	read -p "Neighbour file exists - do you want to rebuild? (Y/N): " run_neighbours
    fi
    # check if needing to run
    if [ "${run_neighbours}" == "Y" ] || [ "${run_neighbours}" == "y" ]; then
	echo "Running neighbour finding routine"
	# module load conda
	conda activate glamod_QC
    python "${cwd}/find_neighbours.py"

	wc -l "${ROOTDIR}${CONFIG_DIR}${VERSION}neighbours.txt"
    else
	if [ ! -f "${ROOTDIR}${CONFIG_DIR}${VERSION}neighbours.txt" ]; then
	    echo "Not running neighbour finding routine and doesn't exist: Exit"
	    exit
	fi
    fi
fi

# set up list of stations
STATION_LIST="$(grep "station_list " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
station_list_file="${STATION_LIST}"

wc -l "${station_list_file}"
stn_ids=$(awk -F" " '{print $1}' "${station_list_file}")

#**************************************
echo "Check all upstream stations present"
missing_file="${ROOTDIR}${CONFIG_DIR}${VERSION}missing_${STAGE}.txt"
if [ -e "${missing_file}" ]; then
    rm "${missing_file}"
fi
touch "${missing_file}"
for stn in ${stn_ids}
do
    processed=false
    if [ "${STAGE}" == "I" ]; then
        if [ -f "${MFF_DIR}${MFF_VER}${stn}${IN_SUFFIX}${MFF_ZIP}" ]; then
            processed=true
        fi
    elif [ "${STAGE}" == "N" ]; then
        if [ -f "${ROOTDIR}${PROC_DIR}${VERSION}${stn}${OUT_SUFFIX}${QFF_ZIP}" ]; then
            processed=true
        elif [ -f "${ROOTDIR}${QFF_DIR}${VERSION}bad_stations/${stn}${OUT_SUFFIX}${QFF_ZIP}" ]; then
            # if station not processed/withheld, then has been processed, and won't appear
            processed=true
        elif [ -f "${ROOTDIR}${ERR_DIR}${VERSION}${stn}_int.err" ]; then
            # if station has had an error, then has been processed, and won't appear
            processed=true
        fi
    fi

    if [ ${processed} == false ]; then
        echo "${stn}" >> "${missing_file}"
    fi

done

if [ "${STAGE}" == "N" ]; then
    echo "${ROOTDIR}${PROC_DIR}${VERSION}*${OUT_SUFFIX}${QFF_ZIP}"
    n_processed_successfully=$(eval ls "${ROOTDIR}${PROC_DIR}${VERSION}" | wc -l)
    echo "Internal checks successful on ${n_processed_successfully} stations"
    n_processed_bad=$(eval ls "${ROOTDIR}${QFF_DIR}${VERSION}bad_stations/*${OUT_SUFFIX}${QFF_ZIP}" | wc -l)
    echo "Internal checks withheld ${n_processed_bad} stations"
    n_processed_err=$(eval ls "${ROOTDIR}${ERR_DIR}${VERSION}*int.err" | wc -l)
    echo "Internal checks had errors on ${n_processed_err} stations"
fi

echo "Checked for all input files - see missing.txt"
n_missing=$(wc "${missing_file}" | awk -F' ' '{print $1}')
if [ "${n_missing}" -ne 0 ]; then
    read -p "${n_missing} upstream files missing - do you want to run remainder Y/N? " run_bastion
    if [ "${run_bastion}" == "N" ] || [ "${run_bastion}" == "n" ]; then
        exit
    fi
else
    read -p "All upstream files present - do you want to run the job Y/N? " run_bastion
    if [ "${run_bastion}" == "N" ] || [ "${run_bastion}" == "n" ]; then
        exit
    fi

fi

#**************************************
# Batch the stations into a number of parallel scripts
#   one script per $STATIONS_PER_BATCH stations
batch=1
parallel_script="$(prepare_parallel_script "${batch}")"

#**************************************
# spin through each in turn, creating a job
scnt=1
for stn in ${stn_ids}
do
    echo "${stn}"

    # check target file exists (in case waiting on upstream process)
    submit=false
    while [ ${submit} == false ];
    do
    # check if upstream data files are present
	if [ "${STAGE}" == "I" ]; then
        if [ -f "${MFF_DIR}${MFF_VER}${stn}${IN_SUFFIX}${MFF_ZIP}" ]; then
		    submit=true
        fi
	elif [ "${STAGE}" == "N" ]; then
        if [ -f "${ROOTDIR}${PROC_DIR}${VERSION}${stn}${OUT_SUFFIX}${QFF_ZIP}" ]; then
		    submit=true
        elif [ -f "${ROOTDIR}${QFF_DIR}${VERSION}bad_stations/${stn}${OUT_SUFFIX}${QFF_ZIP}" ]; then
		    # if station not processed, then no point submitting
		    submit=false
        elif [ -f "${ROOTDIR}${ERR_DIR}${VERSION}${stn}_int.err" ]; then
		    # if station has had an error, then no point in submitting
		    submit=false
#            else
#                # file may well have been withheld, so skip for the moment
#                # 2020-06-01 - needs to be sorted better (checking the bad_stations folder)
#                submit=true
        fi
	fi

	# option to skip over if upstream missing through unexpected way
	if [ "${WAIT}" == "T" ]; then
        if [ ${submit} == false ]; then
		    echo "upstream file ${stn} missing, sleeping 1m"
		    sleep 1m
        fi

	elif [ "${WAIT}" == "F" ]; then
        if [ ${submit} == false ]; then
		    echo "upstream file ${stn} missing, skipping"
		    break
	        # to escape the loop as we will skip this file
        fi
	fi
    done

    # Have upstream file indicator, so can now insert into script
    # make the Parallel script and submit
    if [ ${submit} == true ]; then

        if [ "${STAGE}" == "I" ]; then
	        if [ ! -e "${ROOTDIR}${PROC_DIR}${VERSION}" ]; then
		        mkdir "${ROOTDIR}${PROC_DIR}${VERSION}"
	        fi
	    elif [ "${STAGE}" == "N" ]; then
	        if [ ! -e "${ROOTDIR}${QFF_DIR}${VERSION}" ]; then
		        mkdir "${ROOTDIR}${QFF_DIR}${VERSION}"
	        fi
    	fi

        # if overwrite
        if [ "${CLOBBER}" == "C" ]; then

	        if [ "${STAGE}" == "I" ]; then
		        echo "python3 ${cwd}/intra_checks.py --restart_id ${stn} --end_id ${stn} --full --clobber" >> "${parallel_script}"
	        elif  [ "${STAGE}" == "N" ]; then
		        echo "python3 ${cwd}/inter_checks.py --restart_id ${stn} --end_id ${stn} --full --clobber" >> "${parallel_script}"
	        fi
            # increment station counter (don't for other elifs to reduce jobs)
            let scnt=scnt+1

	    # if not overwrite
	    else

            # check if already processed before setting going
            if [ "${STAGE}" == "I" ]; then

                if [ -f "${ROOTDIR}${PROC_DIR}${VERSION}${stn}${OUT_SUFFIX}${QFF_ZIP}" ]; then
                    # output exists
                    echo "${stn} already processed"

                elif [ -f "${ROOTDIR}${QFF_DIR}${VERSION}bad_stations/${stn}${OUT_SUFFIX}${QFF_ZIP}" ]; then
                    # output exists
                    echo "${stn} already processed - bad station"

                elif [ -f "${ROOTDIR}${ERR_DIR}${VERSION}${stn}_int.err" ]; then
                    # output exists
                    echo "${stn} already processed - managed error"

                else

		            # no output, include
		            echo "python3 ${cwd}/intra_checks.py --restart_id ${stn} --end_id ${stn} --full" >> "${parallel_script}"

                    # increment station counter (don't for other elifs to reduce jobs)
                    let scnt=scnt+1
                fi

            elif [ "${STAGE}" == "N" ]; then

                if [ -f "${ROOTDIR}${QFF_DIR}${VERSION}${stn}${OUT_SUFFIX}${QFF_ZIP}" ]; then
                    # output exists
                    echo "${stn} already processed"

                elif [ -f "${ROOTDIR}${QFF_DIR}${VERSION}bad_stations/${stn}${OUT_SUFFIX}${QFF_ZIP}" ]; then
                    # output exists
                    echo "${stn} already processed - bad station"

                elif [ -f "${ROOTDIR}${ERR_DIR}${VERSION}${stn}_ext.err" ]; then
                    # output exists
                    echo "${stn} already processed - managed error"

                else
		            # no output, include
                    echo "python3 ${cwd}/inter_checks.py --restart_id ${stn} --end_id ${stn} --full" >> "${parallel_script}"

                    # increment station counter (don't for other elifs to reduce jobs)
                    let scnt=scnt+1

                fi
	        fi # stage
	    fi # clobber

    else
	    echo "${stn} not submitted, upstream file not available"
    fi # submit

    # and write script to run this batch
    if [ ${scnt} -eq ${STATIONS_PER_BATCH} ]; then
	    write_and_submit_bastion_script "${parallel_script}" "${batch}"

	    # and reset counters and scripts
	    let batch=batch+1
	    parallel_script="$(prepare_parallel_script "${batch}")"
	    scnt=1

	    # just for ease of reading the script output
	    sleep 1
#	    exit

    fi
#    exit

done
# and submit the final batch of stations.
write_and_submit_bastion_script "${parallel_script}" "${batch}"


#**************************************
# and print summary
#n_jobs=$(squeue --user="${USER}" | wc -l)
# deal with Slurm header in output
#let n_jobs=n_jobs-1
#while [ ${n_jobs} -ne 0 ];
#do
#    echo "All submitted, waiting 5min for queue to clear"
#    sleep 5m
#    n_jobs=$(squeue --user="${USER}" | wc -l)
#    let n_jobs=n_jobs-1
#done

source check_if_processed.bash "${STAGE}"

echo "ends"
