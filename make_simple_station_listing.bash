#!/bin/bash
#set -x
set -euo pipefail
#******************************************************************
# Script to produce very simple summary listing of available stations
#******************************************************************


CONFIG_FILE="$(pwd)/configuration.txt"

# VENVDIR="$(grep "venvdir " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
# using spaces after setting ID to ensure pull out correct line
# these are fixed references
ROOTDIR="$(grep "root " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"

# extract remaining locations
MFF_DIR="$(grep "mff " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
MFF_VER="$(grep "mff_version " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
MFF_ZIP="$(grep "in_compression " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
IN_SUFFIX="$(grep "in_suffix " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
VERSION="$(grep "version " "${CONFIG_FILE}" | awk -F'= ' 'FNR == 2 {print $2}')"

# set up list of stations
STATION_LIST="$(grep "station_list " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')"
station_list_file="${STATION_LIST}"

wc -l "${station_list_file}"

if [ "${station_list_file: -4}" == ".txt" ]; then
    # if fixed width format station list
    stn_ids=$(awk -F" " '{print $1}' "${station_list_file}")
elif [ "${station_list_file: -4}" == ".csv" ]; then
    # if comma separted format station list
    stn_ids=$(awk -F"," '{print $1}' "${station_list_file}")
else
    echo "Unknown station list file type.  Expecting fixed width (.txt) or comma separated (.csv)"
fi


OUTFILE="${VERSION::-1}_simple_summary.txt"
printf "%14s\t%10s\t%8s\t%8s\t%8s\t%8s\t%8s\t%8s \n" "GHCN ID" "N Record" "start Y" "start M" "start D" "end Y" "end M" "end D" > ${OUTFILE}

for stn in ${stn_ids}
do

    station_file="${MFF_DIR}${MFF_VER}${stn}${IN_SUFFIX}${MFF_ZIP}"
    if [ -f "${MFF_DIR}${MFF_VER}${stn}${IN_SUFFIX}${MFF_ZIP}" ]; then

        if [ "${MFF_ZIP}" != "" ]; then
            # need to unzip the file
            if [ "${MFF_ZIP}" == ".gz" ]; then
                gunzip "${station_file}"
                station_file="${MFF_DIR}${MFF_VER}${stn}${IN_SUFFIX}"
            fi
        fi

        start_year=$(head -2 ${station_file} | tail -1 | awk -F'|' '{print int($4)}')
        start_mnth=$(head -2 ${station_file} | tail -1 | awk -F'|' '{print int($5)}')
        start_day=$(head -2 ${station_file} | tail -1 | awk -F'|' '{print int($6)}')


        end_year=$(tail -2 ${station_file} | head -1 | awk -F'|' '{print int($4)}')
        end_mnth=$(tail -2 ${station_file} | head -1 | awk -F'|' '{print int($5)}')
        end_day=$(tail -2 ${station_file} | head -1 | awk -F'|' '{print int($6)}')

        n_records=$(wc -l ${station_file} | awk -F' ' '{print $1}')
        echo ${stn} ${n_records} ${start_year} ${start_mnth} ${start_day} ${end_year} ${end_mnth} ${end_day}
        printf "%14s\t%10d\t%8d\t%8d\t%8d\t%8d\t%8d\t%8d \n" ${stn} ${n_records} ${start_year} ${start_mnth} ${start_day} ${end_year} ${end_mnth} ${end_day} >> ${OUTFILE}

        if [ "${MFF_ZIP}" != "" ]; then
            # need to re-zip the file
            if [ "${MFF_ZIP}" == ".gz" ]; then
                gzip "${station_file}"
            fi
        fi

    fi
done