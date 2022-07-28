#!/bin/bash
#set -x
#****************************************************************** 
# Script to cross check after each run to find all outputs and double
#    check that everything has run
#
# CALL
#    bash check_if_processed.bash STAGE
#    
#    STAGE = I [internal] or N [neighbour]
#****************************************************************** 
STAGE=$1
if [ "${STAGE}" != "I" ] && [ "${STAGE}" != "N" ]; then
    echo Please enter valid switch. I [internal] or N [neighbour]
    exit
fi


cwd=`pwd`

# use configuration file to pull out paths &c
CONFIG_FILE="${cwd}/configuration.txt"

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


# set up list of stations
STATION_LIST=$(grep "station_list " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
station_list_file="${STATION_LIST}"
#echo $station_list_file
echo `wc -l ${station_list_file}`
stn_ids=`awk -F" " '{print $1}' ${station_list_file}`

# set up lists
processed=0
withheld=0
errors=0
unprocessed=0


# spin through each in turn, submitting a job
scnt=0
for stn in ${stn_ids}
do
#    echo ${stn}

    if [ "${STAGE}" == "I" ]; then
        process_dir=${ROOTDIR}${PROC}${VERSION}
    elif [ "${STAGE}" == "N" ]; then
        process_dir=${ROOTDIR}${QFF}${VERSION}
    fi
    withheld_dir=${ROOTDIR}${QFF}${VERSION}bad_stations
    error_dir=${ROOTDIR}${ERR}${VERSION}

    if [ "${STAGE}" == "I" ]; then
        if [ -f "${ROOTDIR}${PROC}${VERSION}${stn}.qff" ]; then
            # internal checks completed
            let processed=processed+1
        elif [ -f "${ROOTDIR}${QFF}${VERSION}bad_stations/${stn}.qff" ]; then
            # internal checks led to station being withheld
            let withheld=withheld+1
        elif [ -f "${ROOTDIR}${ERR}${VERSION}${stn}.err" ]; then
            # internal checks led to station being withheld
            let errors=errors+1
        else
            # this shouldn't happen!
            echo "${stn} missing"
            let unprocessed=unprocessed+1
        fi

    elif [ "${STAGE}" == "N" ]; then
        if [ -f "${ROOTDIR}${QFF}${VERSION}${stn}.qff" ]; then
            # external checks completed
            let processed=processed+1
        elif [ -f "${ROOTDIR}${QFF}${VERSION}bad_stations/${stn}.qff" ]; then
            # internal checks led to station being withheld
            let withheld=withheld+1
        elif [ -f "${ROOTDIR}${ERR}${VERSION}${stn}.err" ]; then
            # internal checks led to station being withheld
            let errors=errors+1
        else
            # this shouldn't happen!
            echo "${stn} missing"
            let unprocessed=unprocessed+1
        fi
    fi

#    echo $processed $withheld $errors
    
done

in_stations=`wc -l ${station_list_file}`
echo "Total input stations ${in_stations}" 

echo "Total qc'd stations ${processed} ${process_dir}"

echo "Total withheld stations ${withheld} ${withheld_dir}"

echo "Total errors ${errors} ${error_dir}"

let out_stations=processed+withheld+errors
echo "Total output stations ${out_stations}"
echo ""
missing=$(wc missing.txt | awk -F' ' '{print $1}')
echo "Upstream missing stations ${missing}"
let unprocessed=unprocessed-missing
echo "Unprocessed stations (job failures?) ${unprocessed}"

let out_stations=processed+withheld+errors+unprocessed
echo "Total stations (excl upstream missing) ${out_stations}"
