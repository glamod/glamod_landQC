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



# use configuration file to pull out paths &c
CONFIG_FILE="$(pwd)/configuration.txt"

# using spaces after setting ID to ensure pull out correct line
# these are fixed references
ROOTDIR=$(grep "root " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')

# extract remaining locations [same construct as in run scripts]
# MFF=$(grep "mff " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
# MFF_VER=$(grep "mff_version " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
#MFF_ZIP="$(grep "in_compression " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
PROC_DIR=$(grep "proc " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
QFF_DIR=$(grep "qff " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
QFF_ZIP="$(grep "out_compression " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
VERSION=$(grep "version " "${CONFIG_FILE}" | awk -F'= ' 'FNR == 2 {print $2}')
ERR_DIR="$(grep "errors " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
CONFIG_DIR="$(grep "config " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
# IN_SUFFIX="$(grep "in_suffix " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
OUT_SUFFIX="$(grep "out_suffix " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"

# set up list of stations
STATION_LIST=$(grep "station_list " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')
station_list_file="${STATION_LIST}"
#echo $station_list_file
wc -l "${station_list_file}"
stn_ids=$(awk -F" " '{print $1}' "${station_list_file}")

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
        process_dir=${ROOTDIR}${PROC_DIR}${VERSION}
    elif [ "${STAGE}" == "N" ]; then
        process_dir=${ROOTDIR}${QFF_DIR}${VERSION}
    fi
    withheld_dir=${ROOTDIR}${QFF_DIR}${VERSION}bad_stations
    error_dir=${ROOTDIR}${ERR_DIR}${VERSION}


    if [ "${STAGE}" == "I" ]; then
        if [ -f "${ROOTDIR}${PROC_DIR}${VERSION}${stn}${OUT_SUFFIX}${QFF_ZIP}" ]; then
            # internal checks completed
            (( processed=processed+1 ))
        elif [ -f "${ROOTDIR}${QFF_DIR}${VERSION}bad_stations/${stn}${OUT_SUFFIX}${QFF_ZIP}" ]; then
            # internal checks led to station being withheld
            (( withheld=withheld+1 ))
        elif [ -f "${ROOTDIR}${ERR_DIR}${VERSION}${stn}.err" ]; then
            # internal checks led to station being withheld
            (( errors=errors+1 ))
        else
            # this shouldn't happen!
            echo "${stn} missing"
            (( unprocessed=unprocessed+1 ))
        fi

    elif [ "${STAGE}" == "N" ]; then
        if [ -f "${ROOTDIR}${QFF_DIR}${VERSION}${stn}${OUT_SUFFIX}${QFF_ZIP}" ]; then
            # external checks completed
            (( processed=processed+1 ))
        elif [ -f "${ROOTDIR}${QFF_DIR}${VERSION}bad_stations/${stn}${OUT_SUFFIX}${QFF_ZIP}" ]; then
            # internal checks led to station being withheld
            (( withheld=withheld+1 ))
        elif [ -f "${ROOTDIR}${ERR_DIR}${VERSION}${stn}.err" ]; then
            # internal checks led to station being withheld
            (( errors=errors+1 ))
        else
            # this shouldn't happen!
            echo "${stn} missing"
            (( unprocessed=unprocessed+1 ))
        fi
    fi

#    echo $processed $withheld $errors

done

in_stations=$(wc -l ${station_list_file})
echo "Total input stations ${in_stations}"

echo "Total qc'd stations ${processed} ${process_dir}"

echo "Total withheld stations ${withheld} ${withheld_dir}"

echo "Total errors ${errors} ${error_dir}"

(( out_stations=processed+withheld+errors ))
echo "Total output stations ${out_stations}"
echo ""
missing_file="${ROOTDIR}${CONFIG_DIR}${VERSION}missing_${STAGE}.txt"
missing=$(wc "${missing_file}" | awk -F' ' '{print $1}')
echo "Upstream missing stations ${missing}"
(( unprocessed=unprocessed-missing ))
echo "Unprocessed stations (job failures?) ${unprocessed}"

(( out_stations=processed+withheld+errors+unprocessed ))
echo "Total stations (excl upstream missing) ${out_stations}"
