#!/usr/bin/env python
'''
initial_setup - contains settings and path information
'''


#*********************************************
import os
import configparser
import sys
import json
import numpy as np
import stat

#*********************************************
# Process Configuration file
CONFIG_FILE = "./configuration.txt"

if not os.path.exists(os.path.join(os.path.dirname(__file__), CONFIG_FILE)):
    print("Configuration file missing - {}".format(os.path.join(os.path.dirname(__file__), CONFIG_FILE)))
    sys.exit()
else:
    CONFIG_FILE = os.path.join(os.path.dirname(__file__), CONFIG_FILE)


# read in configuration file
config = configparser.ConfigParser()
config.read(CONFIG_FILE)

#*********************************************
# locations

# source always the same - currently on GWS
SUBDAILY_MINGLE_DIR = config.get("PATHS", "mff")
SUBDAILY_MFF_DIR = os.path.join(SUBDAILY_MINGLE_DIR, config.get("PATHS", "mff_version"))

# base dir to run on - GWS or scratch
ROOT_DIR = config.get("PATHS", "root")
DATESTAMP = config.get("PATHS", "version")


# set up suitable paths
# processing space - for intermediate files, between mff and qff
if not os.path.exists(os.path.join(ROOT_DIR, config.get("PATHS", "proc"))):
    os.makedirs(os.path.join(ROOT_DIR, config.get("PATHS", "proc")))

SUBDAILY_PROC_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "proc"), DATESTAMP)
if not os.path.exists(SUBDAILY_PROC_DIR):
    os.makedirs(SUBDAILY_PROC_DIR)

if not os.path.exists(os.path.join(ROOT_DIR, config.get("PATHS", "qff"))):
    os.mkdir(os.path.join(ROOT_DIR, config.get("PATHS", "qff")))

SUBDAILY_OUT_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "qff"), DATESTAMP)
if not os.path.exists(SUBDAILY_OUT_DIR):
    os.makedirs(SUBDAILY_OUT_DIR)
#    os.chmod(SUBDAILY_OUT_DIR, stat.S_IWGRP)

SUBDAILY_BAD_DIR = os.path.join(SUBDAILY_OUT_DIR, "bad_stations") #  datestamp in Subdaily_out_dir
if not os.path.exists(SUBDAILY_BAD_DIR):
    os.makedirs(SUBDAILY_BAD_DIR)

SUBDAILY_CONFIG_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "config"), DATESTAMP)
if not os.path.exists(SUBDAILY_CONFIG_DIR):
    os.makedirs(SUBDAILY_CONFIG_DIR)

SUBDAILY_IMAGE_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "images"), DATESTAMP)
if not os.path.exists(SUBDAILY_IMAGE_DIR):
    os.makedirs(SUBDAILY_IMAGE_DIR)

SUBDAILY_FLAG_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "flags"), DATESTAMP)
if not os.path.exists(SUBDAILY_FLAG_DIR):
    os.makedirs(SUBDAILY_FLAG_DIR)

SUBDAILY_ERROR_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "errors"), DATESTAMP)
if not os.path.exists(SUBDAILY_ERROR_DIR):
    os.makedirs(SUBDAILY_ERROR_DIR)


# name of station list
STATION_LIST = config.get("FILES", "station_list")

# for cross-timescale checks
# DAILY_DIR = 
# MONTHLY_DIR = 

#*********************************************
# read in parameter list
VARFILE = config.get("FILES", "variables")
with open(os.path.join(os.path.dirname(__file__), VARFILE), "r") as pf:
    parameters = json.load(pf)
obs_var_list = parameters["variables"]["process_vars"]
carry_thru_var_list = parameters["variables"]["not_process_vars"]

DTYPE_DICT =  {}
for var_list in (obs_var_list, carry_thru_var_list):
    for v, var in enumerate(var_list):
        if var in ["remarks", "pressure_3hr_change"]:
            DTYPE_DICT[var] = str
        elif "pres_wx" in var:
            # catch all present weather codes (which may contain character info)
            DTYPE_DICT[var] = str
        else:
            DTYPE_DICT[var] = np.float64

        DTYPE_DICT["{}_Source_ID".format(var)] = str
        DTYPE_DICT["{}_QC_flag".format(var)] = str
        DTYPE_DICT["{}_Measurement_Code".format(var)] = str
        DTYPE_DICT["{}_Quality_Code".format(var)] = str
        DTYPE_DICT["{}_Report_Type".format(var)] = str
        DTYPE_DICT["{}_Source_Code".format(var)] = str
        DTYPE_DICT["{}_Source_Station_ID".format(var)] = str
#        DTYPE_DICT["Source_ID.{}".format(v+1)] = str

DTYPE_DICT["Source_ID"] = str



#************************************************************************
if __name__ == "__main__":

    # if called as stand alone, then clean out the errors from a previous run
    if len(os.listdir(SUBDAILY_ERROR_DIR)) != 0:
        for this_file in os.listdir(SUBDAILY_ERROR_DIR):
            os.remove(os.path.join(SUBDAILY_ERROR_DIR, this_file))
        

#*********************************************
# END
#*********************************************
