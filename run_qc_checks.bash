#!/bin/bash
set -x
#******************************************************************
# Script to process all the stations.  Runs through station list
#   and submits each as a separate jobs to SPICE
#
# CALL
#    bash run_qc.bash STAGE WAIT CLOBBER
#
#    STAGE = I [internal] or N [neighbour]
#     WAIT = T [true] or F [false] # wait for upstream files to be ready
#  CLOBBER = C [clobber] or S [skip] # overwrite or skip existing files
#******************************************************************

STAGE=$1
if [ "${STAGE}" != "I" ] && [ "${STAGE}" != "N" ]; then
    echo "Please enter valid switch. I [internal] or N [neighbour]"
    exit
fi
WAIT=$2
if [ "${WAIT}" != "T" ] && [ "${WAIT}" != "F" ]; then
    echo "Please enter valid waiting option. T [true - wait for upstream files] or F [false - skip missing files]"
    exit
fi
CLOBBER=$3
if [ "${CLOBBER}" != "C" ] && [ "${CLOBBER}" != "S" ]; then
    echo "Please enter valid clobber option. C [clobber - overwrite existing outputs] or S [skip - keep existing outputs]"
    exit
fi
# remove all 3 positional characters
shift
shift
shift

#**************************************
# other settings
if [ "${STAGE}" == "I" ]; then
    MAX_N_JOBS=150
elif [ "${STAGE}" == "N" ]; then
    MAX_N_JOBS=50
fi
WAIT_N_MINS=1
cwd=$(pwd)

SCRIPT_DIR=${cwd}/spice_scripts/
if [ ! -d "${SCRIPT_DIR}" ]; then
    mkdir "${SCRIPT_DIR}"
fi

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
ERR_DIR="$(grep "errors " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"  # ${QFF_DIR%/}_errors/
LOG_DIR="$(grep "logs " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
CONFIG_DIR="$(grep "config " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
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
    conda init bash > /dev/null 2>&1
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
STATION_LIST=$(grep "station_list " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
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
        if [ -f "${MFF_DIR}${MFF_VER}${stn}.mff${MFF_ZIP}" ]; then
            processed=true
        fi
    elif [ "${STAGE}" == "N" ]; then
        if [ -f "${ROOTDIR}${PROC_DIR}${VERSION}${stn}.qff${QFF_ZIP}" ]; then
            processed=true
        elif [ -f "${ROOTDIR}${QFF_DIR}${VERSION}bad_stations/${stn}.qff${QFF_ZIP}" ]; then
            # if station not processed, then has been processed, and won't appear
            processed=true
        elif [ -f "${ROOTDIR}${ERR_DIR}${VERSION}${stn}.err" ]; then
            # if station has had an error, then has been processed, and won't appear
            processed=true
        fi
    fi

    if [ ${processed} == false ]; then
        echo "${stn}" >> "${missing_file}"
    fi

done

if [ "${STAGE}" == "N" ]; then
    echo "${ROOTDIR}${PROC_DIR}${VERSION}*.qff${QFF_ZIP}"
    n_processed_successfully=$(eval ls "${ROOTDIR}${PROC_DIR}${VERSION}" | wc -l)
    echo "Internal checks successful on ${n_processed_successfully} stations"
    n_processed_bad=$(eval ls "${ROOTDIR}${QFF_DIR}${VERSION}bad_stations/*.qff${QFF_ZIP}" | wc -l)
    echo "Internal checks withheld ${n_processed_bad} stations"
    n_processed_err=$(eval ls "${ROOTDIR}${ERR_DIR}${VERSION}/*err" | wc -l)
    echo "Internal checks had errors on ${n_processed_err} stations"
fi

echo "Checked for all input files - see missing.txt"
n_missing=$(wc "${missing_file}" | awk -F' ' '{print $1}')
if [ "${n_missing}" -ne 0 ]; then
    read -p "${n_missing} upstream files missing - do you want to run remainder Y/N? " run_spice
    if [ "${run_spice}" == "N" ]; then
        exit
    fi
else
    read -p "All upstream files present - do you want to run the job Y/N? " run_spice
    if [ "${run_spice}" == "N" ] || [ "${run_spice}" == "n" ]; then
        exit
    fi

fi


#**************************************
# spin through each in turn, submitting a job
# Mix up the stations, so that not all the big/long ones (USA etc)
#   Are in the same jobs
shuffled_stns=$(echo "${stn_ids}" | xargs shuf -e)

scnt=0
for stn in ${shuffled_stns}
do
    echo "${stn}"

    # make the SPICE script and submit
    if [ "${STAGE}" == "I" ]; then
 	    spice_script="${SCRIPT_DIR}/spice_internal_${stn}.bash"
    elif  [ "${STAGE}" == "N" ]; then
 	    spice_script="${SCRIPT_DIR}/spice_external_${stn}.bash"
    fi
    echo "#!/bin/bash -l" > "${spice_script}"
    # ICHEC settings
    # echo "#SBATCH --partition=short-serial-4hr" >> "${spice_script}"
    # echo "#SBATCH --account=short4hr" >> "${spice_script}"
    # SPICE settings
    echo "#SBATCH --qos=normal" >> "${spice_script}"

    echo "#SBATCH --job-name=QC_${stn}" >> "${spice_script}"
    echo "#SBATCH --output=${ROOTDIR}${LOG_DIR}/${stn}_${STAGE}.out" >> "${spice_script}"
    echo "#SBATCH --error=${ROOTDIR}${LOG_DIR}/${stn}_${STAGE}.err " >> "${spice_script}"

    if [ "${STAGE}" == "I" ]; then
        if [ "${stn:0:1}" == "U" ]; then
            # US stations take a long time
            echo "#SBATCH --time=60:00" >> "${spice_script}" # 60mins
            echo "#SBATCH --mem=15000" >> "${spice_script}"
        elif [ "${stn:0:1}" == "G" ]; then
            # Some German stations take a long time
            echo "#SBATCH --time=60:00" >> "${spice_script}" # 60mins
            echo "#SBATCH --mem=12000" >> "${spice_script}"
        else
            echo "#SBATCH --time=30:00" >> "${spice_script}" # 20mins
            echo "#SBATCH --mem=8000" >> "${spice_script}"
        fi
    elif  [ "${STAGE}" == "N" ]; then
        if [ "${stn:0:1}" == "U" ]; then
            # US stations take lots of memory
            echo "#SBATCH --time=20:00" >> "${spice_script}" # 20mins
            echo "#SBATCH --mem=30000" >> "${spice_script}"
        elif [ "${stn:0:1}" == "G" ]; then
            # Some German stations take lots of memory
            echo "#SBATCH --time=20:00" >> "${spice_script}" # 20mins
            echo "#SBATCH --mem=30000" >> "${spice_script}"
        else
            echo "#SBATCH --time=20:00" >> "${spice_script}" # 20mins
            echo "#SBATCH --mem=10000" >> "${spice_script}"
        fi
    fi
    echo "" >> "${spice_script}"
    # echo "source ${VENVDIR}/bin/activate" >> "${spice_script}"
    echo "conda activate glamod_QC" >> "${spice_script}"
    echo "" >> "${spice_script}"

    if [ "${STAGE}" == "I" ]; then
        echo "python -m intra_checks --restart_id ${stn} --end_id ${stn} --clobber --full" >> "${spice_script}"
    elif  [ "${STAGE}" == "N" ]; then
        echo "python -m inter_checks --restart_id ${stn} --end_id ${stn} --clobber --full" >> "${spice_script}"
    fi

    # now check if we should submit it.
    # ensure don't overload the queue, max of e.g. 50
    n_jobs=$(squeue --user="${USER}" | wc -l)
    while [ "${n_jobs}" -gt "${MAX_N_JOBS}" ];
    do
        echo "sleeping for ${WAIT_N_MINS}min to clear queue"
        sleep "${WAIT_N_MINS}m"
        n_jobs=$(squeue --user="${USER}" | wc -l)
    done

    let scnt=scnt+1
# SUBSETTING FOR DIAGNOSTICS
#   if [ ${stn} == "AYW00057801" ]; then
#       # test first 1000 (20240704)
#       exit
#   fi
#    if [ ${scnt} -le 2000 ]; then
#        # test first 1000 (20190913)
#        continue
#    elif [ ${scnt} -ge 4000 ]; then
#        exit
#    fi

    # check target file exists (in case waiting on upstream process)
    submit=false
    while [ ${submit} == false ];
    do
        if [ "${STAGE}" == "I" ]; then
            if [ -f "${MFF_DIR}${MFF_VER}${stn}.mff${MFF_ZIP}" ]; then
                submit=true
            fi
        elif [ "${STAGE}" == "N" ]; then
            if [ -f "${ROOTDIR}${PROC_DIR}${VERSION}${stn}.qff${QFF_ZIP}" ]; then
                submit=true
            elif [ -f "${ROOTDIR}${QFF_DIR}${VERSION}bad_stations/${stn}.qff${QFF_ZIP}" ]; then
                # if station not processed, then no point submitting
                submit=false
            elif [ -f "${ROOTDIR}${ERR_DIR}${VERSION}${stn}.err" ]; then
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
                submit=skip # to escape the loop as we will skip this file
            fi
        fi

    done

    # if clear to submit
    if [ ${submit} == true ]; then

        # make directories if they don't exist
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
            sbatch "${spice_script}"
            sleep 1s # allow submission to occur before moving on

	    # if not overwrite
        else
            # check if already processed before setting going
            if [ "${STAGE}" == "I" ]; then

                if [ -f "${ROOTDIR}${PROC_DIR}${VERSION}${stn}.qff${QFF_ZIP}" ]; then
                    # output exists
                    echo "${stn} already processed"

                elif [ -f "${ROOTDIR}${QFF_DIR}${VERSION}bad_stations/${stn}.qff${QFF_ZIP}" ]; then
                    # output exists
                    echo "${stn} already processed - bad station"

                elif [ -f "${ROOTDIR}${ERR_DIR}${VERSION}${stn}.err" ]; then
                    # output exists
                    echo "${stn} already processed - managed error"

                else
		            # no output, submit
                    sbatch "${spice_script}"
                    sleep 1s # allow submission to occur before
                    # exit

                fi

            elif [ "${STAGE}" == "N" ]; then

                if [ -f "${ROOTDIR}${QFF_DIR}${VERSION}${stn}.qff${QFF_ZIP}" ]; then
                    # output exists
                    echo "${stn} already processed"

                elif [ -f "${ROOTDIR}${QFF_DIR}${VERSION}bad_stations/${stn}.qff${QFF_ZIP}" ]; then
                    # output exists
                    echo "${stn} already processed - bad station"

                elif [ -f "${ROOTDIR}${ERR_DIR}${VERSION}${stn}.err" ]; then
                    # output exists
                    echo "${stn} already processed - managed error"

                else
		            # no output, submit
                    sbatch "${spice_script}"
                    sleep 1s # allow submission to occur before
#                    exit

                fi

            fi

        fi

    fi
#    exit

done


#**************************************
# and print summary
n_jobs=$(squeue --user="${USER}" | wc -l)
# deal with Slurm header in output
let n_jobs=n_jobs-1
while [ ${n_jobs} -ne 0 ];
do
    echo "All submitted, waiting 5min for queue to clear"
    sleep 5m
    n_jobs=$(squeue --user="${USER}" | wc -l)
    let n_jobs=n_jobs-1
done

source check_if_processed.bash "${STAGE}"

echo "ends"
