#!/bin/bash
#set -x
#****************************************************************** 
# Script to process all the stations.  Runs through station list
#   and submits each as a separate jobs to KAY
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
CLOBBER=$3
if [ "${CLOBBER}" != "C" ] && [ "${CLOBBER}" != "S" ]; then
    echo Please enter valid clobber option. C [clobber - overwrite existing outputs] or S [skip - keep existing outputs]
    exit
fi
# remove all 3 positional characters
shift
shift
shift

# other settings
cwd=`pwd`
STATIONS_PER_BATCH=5000

SCRIPT_DIR=${cwd}/taskfarm_scripts/
if [ ! -d ${SCRIPT_DIR} ]; then
    mkdir ${SCRIPT_DIR}
fi

LOG_DIR=${cwd}/taskfarm_logs/
if [ ! -d ${LOG_DIR} ]; then
    mkdir ${LOG_DIR}
fi

#**************************************
# use configuration file to pull out paths &c
CONFIG_FILE="${cwd}/configuration.txt"

VENVDIR=$(grep "venvdir " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
# using spaces after setting ID to ensure pull out correct line
# these are fixed references
ROOTDIR=$(grep "root " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')

# extract remaining locations
MFF=$(grep "mff " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
MFF_VER=$(grep "mff_version " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
PROC=$(grep "proc " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
QFF=$(grep "qff " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
VERSION=$(grep "version " "${CONFIG_FILE}" | grep -v "${MFF_VER}" | awk -F'= ' '{print $2}')
ERR=${QFF%/}_errors/

#**************************************
# if neighbour checks make sure all files in place
if [ "${STAGE}" == "N" ]; then
    echo "${ROOTDIR}${QFF%/}_configs/${VERSION}neighbours.txt"
    if [ ! -f "${ROOTDIR}${QFF%/}_configs/${VERSION}neighbours.txt" ]; then
        read -p "Neighbour file missing - do you want to run Y/N" run_neighbours

	    if [ "${run_neighbours}" == "Y" ]; then
	         echo "Running neighbour finding routine"
                 source ${VENVDIR}/bin/activate
                 python ${cwd}/find_neighbours.py
	    else
	         echo "Not running neighbour finding routine, exit"
	         exit
	    fi
    fi
fi


# set up list of stations
STATION_LIST=$(grep "station_list " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
station_list_file="${STATION_LIST}"

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
        if [ -f "${MFF}${MFF_VER}${stn}.mff" ]; then
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
    read -p "${n_missing} upstream files missing - do you want to run remainder Y/N? " run_kay
    if [ "${run_kay}" == "N" ]; then
        exit
    fi
fi

batch=1
if [ "${STAGE}" == "I" ]; then
    taskfarm_script="${SCRIPT_DIR}/taskfarm_internal_${batch}.bash"
elif  [ "${STAGE}" == "N" ]; then
    taskfarm_script="${SCRIPT_DIR}/taskfarm_external_${batch}.bash"
fi
if [ -e ${taskfarm_script} ]; then
    rm ${taskfarm_script}
fi
#**************************************
# spin through each in turn, creating a job
scnt=1
for stn in ${stn_ids}
do
    echo ${stn}
    
    # check target file exists (in case waiting on upstream process)
    submit=false
    while [ ${submit} == false ];
    do
    # check if upstream data files are present
	if [ "${STAGE}" == "I" ]; then
            if [ -f "${MFF}${MFF_VER}${stn}.mff" ]; then
		submit=true
            fi
	elif [ "${STAGE}" == "N" ]; then
            if [ -f "${ROOTDIR}${PROC}${VERSION}${stn}.qff" ]; then
		submit=true
            elif [ -f "${ROOTDIR}${QFF}${VERSION}bad_stations/${stn}.qff" ]; then
		# if station not processed, then no point submitting
		submit=false
            elif [ -f "${ROOTDIR}${ERR}${VERSION}${stn}.err" ]; then
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
    # make the Taskfarm script and submit
    if [ ${submit} == true ]; then

        if [ "${STAGE}" == "I" ]; then
	    if [ ! -e "${ROOTDIR}${PROC}${VERSION}" ]; then
		mkdir ${ROOTDIR}${PROC}${VERSION}
	    fi
	elif [ "${STAGE}" == "N" ]; then
	    if [ ! -e "${ROOTDIR}${QFF}${VERSION}" ]; then
		mkdir ${ROOTDIR}${QFF}${VERSION}
	    fi
	fi

        # if overwrite
        if [ "${CLOBBER}" == "C" ]; then

	    if [ "${STAGE}" == "I" ]; then
		echo "python3 ${cwd}/intra_checks.py --restart_id ${stn} --end_id ${stn} --full --diagnostics" >> ${taskfarm_script}
	    elif  [ "${STAGE}" == "N" ]; then
		echo "python3 ${cwd}/inter_checks.py --restart_id ${stn} --end_id ${stn} --full --diagnostics" >> ${taskfarm_script}
	    fi

	# if not overwrite
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
		    # no output, include
		    echo "python3 ${cwd}/intra_checks.py --restart_id ${stn} --end_id ${stn} --full --diagnostics" >> ${taskfarm_script}
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
		    # no output, include
                    echo "python3 ${cwd}/inter_checks.py --restart_id ${stn} --end_id ${stn} --full --diagnostics" >> ${taskfarm_script}
                fi

	    fi # stage
	fi # clobber

    else
	echo "${stn} not submitted, upstream file not available"
    fi # submit

    # increment station counter
    let scnt=scnt+1
    
    # batch up 
    if [ ${scnt} -eq ${STATIONS_PER_BATCH} ]; then
	if [ "${STAGE}" == "I" ]; then
 	    kay_script="${SCRIPT_DIR}/kay_internal_${batch}.bash"
	elif  [ "${STAGE}" == "N" ]; then
 	    kay_script="${SCRIPT_DIR}/kay_external_${batch}.bash"
	fi
	
	if [ ! -e ${kay_script} ]; then
	    rm ${kay_script}
	fi
	echo "#!/bin/bash -l" > ${kay_script}
	echo "#SBATCH -p ShmemQ" >> ${kay_script}
	echo "#SBATCH -N 1" >> ${kay_script}
	echo "#SBATCH -t 24:00:00" >> ${kay_script}
	echo "#SBATCH -A glamod" >> ${kay_script}
	echo "#SBATCH -o ${LOG_DIR}/qc_${VERSION::-1}_${STAGE}_batch${batch}.out" >> ${kay_script}
	echo "#SBATCH -e ${LOG_DIR}/qc_${VERSION::-1}_${STAGE}_batch${batch}.err" >> ${kay_script}
	echo "#SBATCH --mail-user=robert.dunn@metoffice.gov.uk" >> ${kay_script}
	echo "#SBATCH --mail-type=BEGIN,END" >> ${kay_script}
	echo "" >> ${kay_script}
	# TODO sort python environment
	echo "# activate python environment" >> ${kay_script}
	echo "source ${VENVDIR}/bin/activate" >> ${kay_script}
	echo "" >> ${kay_script}
	echo "# go to scripts and set taskfarm running" >> ${kay_script}
	echo "cd ${SCRIPT_DIR}" >> ${kay_script}
	echo "module load taskfarm" >> ${kay_script}
	echo "taskfarm ${taskfarm_script}" >> ${kay_script}
	
	sbatch < ${kay_script}

	# and reset counters and scripts
	let batch=batch+1
	scnt=1

#	exit

    fi
#    exit
      
done

exit
#**************************************
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
