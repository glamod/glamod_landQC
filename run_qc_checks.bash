#!/bin/bash
#set -x
#****************************************************************** 
# Script to process all the stations.  Runs through station list
#   and submits each as a separate jobs to LOTUS
#
# CALL
#    bash run_qc.bash STAGE
#    
#    STAGE = I [internal] or N [neighbour]
#****************************************************************** 
STAGE=$1
if [ "${STAGE}" != "I" ] && [ "${STAGE}" != "N" ]; then
    echo Please enter valid switch. I [internal] or N [neighbour]
    exit
fi


CLOBBER="True"
cwd=`pwd`

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
PROC_VER=$(grep "proc_version " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')

QFF=$(grep "qff " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
QFF_VER=$(grep "qff_version " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')

# set up list of stations
STATION_LIST="ghcnh-stations.txt"
station_list_file=${ROOT}${MFF}${STATION_LIST}
echo $station_list_file
stn_ids=`awk -F" " '{print $1}' ${station_list_file}`

# spin through each in turn, submitting a job
scnt=0
for stn in ${stn_ids}
do
    echo ${stn}
    
    # make the LOTUS script and submit
    lotus_script="${cwd}/lotus_internal_${stn}.bsub"
    echo "#!/bin/bash " > ${lotus_script}
    echo "#BSUB -q short-serial" >> ${lotus_script}
    echo "#BSUB -J QC_${stn}" >> ${lotus_script}
#    echo "#BSUB -o ${cwd}/logs/%J.out" >> ${lotus_script}
#    echo "#BSUB -e ${cwd}/logs/%J.err " >> ${lotus_script}
    echo "#BSUB -o ${cwd}/logs/${stn}.out" >> ${lotus_script}
    echo "#BSUB -e ${cwd}/logs/${stn}.err " >> ${lotus_script}
    echo "#BSUB -W 00:30" >> ${lotus_script}
    echo "#BSUB -R \"rusage[mem=1000] select[type==any]\"" >> ${lotus_script}
    echo "#BSUB -M 1000" >> ${lotus_script}
    echo "" >> ${lotus_script}
    echo "source venv/bin/activate" >> ${lotus_script}
    echo "" >> ${lotus_script}

    if [ "${STAGE}" == "I" ]; then
        echo "python -m intra_checks --restart_id ${stn} --end_id ${stn} --full --diagnostics" >> ${lotus_script}
    elif  [ "${STAGE}" == "N" ]; then
        echo "python -m inter_checks --restart_id ${stn} --end_id ${stn} --full --diagnostics" >> ${lotus_script}
    fi

    # now check if we should submit it.
    # ensure don't overload the queue, max of 50
    n_jobs=`bjobs | wc -l`
    while [ ${n_jobs} -gt 50 ];
    do        
        sleep 5m
        n_jobs=`bjobs | wc -l`
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
    if [ "${STAGE}" == "I" ]; then
        if [ -f "${ROOT}${MFF}${MFF_VER}${stn}.mff" ]; then
            submit=true
        fi
    elif [ "${STAGE}" == "N" ]; then
        if [ -f "${ROOTDIR}${PROC}${PROC_VER}${stn}.qff" ]; then
            submit=true
        fi
    fi
    
    # if clear to submi
    if [ $submit == true ]; then
        # if overwrite
        if [ "${CLOBBER}" = "True" ]; then
            bsub < ${lotus_script}
            sleep 1s # allow submission to occur before moving on
        else
            # check if already processed before setting going
            if [ "${STAGE}" == "I" ]; then
                if [ ! -f "${ROOTDIR}${PROC}${PROC_VER}${stn}.qff" ]; then
                    bsub < ${lotus_script}
                    sleep 1s # allow submission to occur before 
                else
                    echo "${stn} already processed"
                fi
                
            elif [ "${STAGE}" == "N" ]; then
                if [ ! -f "${ROOTDIR}${QFF}${QFF_VER}${stn}.qff" ]; then
                    bsub < ${lotus_script}
                    sleep 1s # allow submission to occur before 
                else
                    echo "${stn} already processed"
                fi
            fi
        fi
    fi
done