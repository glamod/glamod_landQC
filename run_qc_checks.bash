#!/bin/bash
#set -x
#****************************************************************** 
# Script to process all the stations.  Runs through station list
#   and submits each as a separate jobs to LOTUS
#
# CALL
#    bash run_qc.bash STAGE WAIT
#    
#    STAGE = I [internal] or N [neighbour]
#     WAIT = T [true] or F [false] # wait for upstream files to be ready
#****************************************************************** 

STAGE=$1
if [ "${STAGE}" != "I" ] && [ "${STAGE}" != "N" ]; then
    echo Please enter valid switch. I [internal] or N [neighbour]
    exit
fi
WAIT=$2
if [ "${WAIT}" != "T" ] && [ "${WAIT}" != "F" ]; then
    echo Please enter valid waiting option. T [true - wait for upstream files] or F [false - skip missing files]
    exit
fi
# remove both positional characters
shift
shift

if [ "${STAGE}" == "I" ]; then
    MAX_N_JOBS=150
elif [ "${STAGE}" == "N" ]; then
    MAX_N_JOBS=50
fi
WAIT_N_MINS=1
CLOBBER="False"
cwd=`pwd`
SCRIPT_DIR=${cwd}/lotus_scripts/

# use configuration file to pull out paths &c
CONFIG_FILE="${cwd}/configuration.txt"

# using spaces after setting ID to ensure pull out correct line
# these are fixed references
ROOT=$(grep "root " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
SCRATCH=$(grep "scratch " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')

# select where output files go
RUNON=$(grep runon "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
# set root dir depending on where QC files end up
if [ "${RUNON}" == "root" ]; then
    ROOTDIR=${ROOT}
elif [ "${RUNON}" == "scratch" ]; then
    ROOTDIR=${SCRATCH}
fi

# extract remaining locations
MFF=$(grep "mff " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
MFF_VER=$(grep "mff_version " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
PROC=$(grep "proc " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
QFF=$(grep "qff " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
VERSION=$(grep "version " "${CONFIG_FILE}" | grep -v "${MFF_VER}" | awk -F'= ' '{print $2}')
ERR=${QFF%/}_errors/

# if neighbour checks make sure all files in place
if [ "${STAGE}" == "N" ]; then
    echo "${ROOTDIR}${QFF%/}_configs/${VERSION}neighbours.txt"
    if [ ! -f "${ROOTDIR}${QFF%/}_configs/${VERSION}neighbours.txt" ]; then
        read -p "Neighbour file missing - do you want to run Y/N" run_neighbours

	    if [ "${run_neighbours}" == "Y" ]; then
	         echo "Running neighbour finding routine"
             source venv/bin/activate
             python -m find_neighbours
	    else
	         echo "Not running neighbour finding routine, exit"
	         exit
	    fi
    fi
fi


# set up list of stations
STATION_LIST=$(grep "station_list " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
station_list_file="${ROOT}/${MFF}/${STATION_LIST}"

echo `wc -l ${station_list_file}`
stn_ids=`awk -F" " '{print $1}' ${station_list_file}`

#**************************************
echo "Check all upstream stations present"
missing_file=missing.txt
rm ${missing_file}
touch ${missing_file}
for stn in ${stn_ids}
do
    processed=false
    if [ "${STAGE}" == "I" ]; then
        if [ -f "${ROOT}${MFF}${MFF_VER}${stn}.mff" ]; then
            processed=true
        fi
    elif [ "${STAGE}" == "N" ]; then
        if [ -f "${ROOTDIR}${PROC}${VERSION}${stn}.qff" ]; then
            processed=true
        elif [ -f "${ROOTDIR}${QFF}${VERSION}bad_stations/${stn}.qff" ]; then
            # if station not processed, then has been processed, and won't appear
            processed=true
        elif [ -f "${ROOTDIR}${ERR}${VERSION}${stn}.err" ]; then
            # if station has had an error, then has been processed, and won't appear
            processed=true
        fi
    fi

    if [ ${processed} == false ]; then
        echo ${stn} >> ${missing_file}
    fi

done

echo "Checked for all input files - see missing.txt"
n_missing=`wc ${missing_file} | awk -F' ' '{print $1}'`
if [ ${n_missing} -ne 0 ]; then
    read -p "${n_missing} upstream files missing - do you want to run remainder Y/N? " run_lotus
    if [ "${run_lotus}" == "N" ]; then
        exit
    fi
fi


#**************************************
# spin through each in turn, submitting a job
# scnt=0
for stn in ${stn_ids}
do
    echo ${stn}
    
    # make the LOTUS script and submit
    if [ "${STAGE}" == "I" ]; then
 	    lotus_script="${SCRIPT_DIR}/lotus_internal_${stn}.bash"
    elif  [ "${STAGE}" == "N" ]; then
 	    lotus_script="${SCRIPT_DIR}/lotus_external_${stn}.bash"
    fi
    echo "#!/bin/bash -l" > ${lotus_script}
#    echo "#SBATCH --partition=short-serial" >> ${lotus_script}
    echo "#SBATCH --partition=short-serial-4hr" >> ${lotus_script}
    echo "#SBATCH --account=short4hr" >> ${lotus_script}
    echo "#SBATCH --job-name=QC_${stn}" >> ${lotus_script}
    echo "#SBATCH --output=${cwd}/logs/${stn}.out" >> ${lotus_script}
    echo "#SBATCH --error=${cwd}/logs/${stn}.err " >> ${lotus_script}
    
    if [ "${STAGE}" == "I" ]; then
        if [ "${stn:0:1}" == "U" ]; then
            # US stations take a long time
            echo "#SBATCH --time=60:00" >> ${lotus_script} # 60mins
        else
            echo "#SBATCH --time=30:00" >> ${lotus_script} # 20mins
        fi
    elif  [ "${STAGE}" == "N" ]; then
        echo "#SBATCH --time=20:00" >> ${lotus_script} # 20mins
    fi
    echo "#SBATCH --mem=3000" >> ${lotus_script}
    echo "" >> ${lotus_script}
#    echo "source venv/bin/activate" >> ${lotus_script}
    echo "source /gws/smf/j04/c3s311a_lot2/rjhd2/landqc_env/bin/activate" >> ${lotus_script}
    echo "" >> ${lotus_script}

    if [ "${STAGE}" == "I" ]; then
        echo "python -m intra_checks --restart_id ${stn} --end_id ${stn} --full --diagnostics" >> ${lotus_script}
    elif  [ "${STAGE}" == "N" ]; then
        echo "python -m inter_checks --restart_id ${stn} --end_id ${stn} --full --diagnostics" >> ${lotus_script}
    fi

    # now check if we should submit it.
    # ensure don't overload the queue, max of e.g. 50
    n_jobs=`squeue --user=rjhd2 | wc -l`
    while [ ${n_jobs} -gt ${MAX_N_JOBS} ];
    do        
        echo "sleeping for ${WAIT_N_MINS}min to clear queue"
        sleep ${WAIT_N_MINS}m
        n_jobs=`squeue --user=rjhd2 | wc -l`
    done

# SUBSETTING FOR DIAGNOSTICS
#    let scnt=scnt+1
#    if [ ${stn} == "BHM00078584" ]; then
#        # test first 1000 (20190913)
#        exit
#    fi
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
            if [ -f "${ROOT}${MFF}${MFF_VER}${stn}.mff" ]; then
                submit=true
            fi
        elif [ "${STAGE}" == "N" ]; then
            if [ -f "${ROOTDIR}${PROC}${VERSION}${stn}.qff" ]; then
                submit=true
            elif [ -f "${ROOTDIR}${QFF}${VERSION}bad_stations/${stn}.qff" ]; then
                # if station not processed, then no point submitting
                submit=skip
            elif [ -f "${ROOTDIR}${ERR}${VERSION}${stn}.err" ]; then
                # if station has had an error, then no point in submitting
                submit=skip
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
    if [ $submit == true ]; then

        # if overwrite
        if [ "${CLOBBER}" == "True" ]; then
            sbatch ${lotus_script}
            sleep 1s # allow submission to occur before moving on

        else

            # check if already processed before setting going
            if [ "${STAGE}" == "I" ]; then

                if [ -f "${ROOTDIR}${PROC}${VERSION}${stn}.qff" ]; then
                    # output exists
                    echo "${stn} already processed"

                elif [ -f "${ROOTDIR}${QFF}${VERSION}bad_stations/${stn}.qff" ]; then
                    # output exists
                    echo "${stn} already processed - bad station"

                elif [ -f "${ROOTDIR}${ERR}${VERSION}${stn}.err" ]; then
                    # output exists
                    echo "${stn} already processed - managed error"

                else
                    sbatch ${lotus_script}
                    sleep 1s # allow submission to occur before 
#                    exit

                fi
                
            elif [ "${STAGE}" == "N" ]; then

                if [ -f "${ROOTDIR}${QFF}${VERSION}${stn}.qff" ]; then
                    # output exists
                    echo "${stn} already processed"

                elif [ -f "${ROOTDIR}${QFF}${VERSION}bad_stations/${stn}.qff" ]; then
                    # output exists
                    echo "${stn} already processed - bad station"

                elif [ -f "${ROOTDIR}${ERR}${VERSION}${stn}.err" ]; then
                    # output exists
                    echo "${stn} already processed - managed error"

                else
                    sbatch ${lotus_script}
                    sleep 1s # allow submission to occur before 
#                    exit

                fi

            fi

        fi

    fi
#    exit

done


# and print summary
n_jobs=`squeue --user=rjhd2 | wc -l`
# deal with Slurm header in output
let n_jobs=n_jobs-1
while [ ${n_jobs} -ne 0 ];
do        
    echo "All submitted, waiting 5min for queue to clear"
    sleep 5m
    n_jobs=`squeue --user=rjhd2 | wc -l`
    let n_jobs=n_jobs-1
done

source check_if_processed.bash ${STAGE}

echo "ends"
