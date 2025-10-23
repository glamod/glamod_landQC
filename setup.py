#!/usr/bin/env python
'''
initial_setup - contains settings and path information
'''


#*********************************************
from pathlib import Path
import configparser
import sys
import json
import numpy as np

#*********************************************
# Process Configuration file
CONFIG_FILE = "./configuration.txt"

CONFIG_FILE = Path(__file__).parent / CONFIG_FILE
if not CONFIG_FILE.exists():
    print(f"Configuration file missing - {CONFIG_FILE}")
    quit()

# read in configuration file
config = configparser.ConfigParser()
config.read(CONFIG_FILE)

#*********************************************
# locations

# source always the same - currently on GWS
SUBDAILY_MINGLE_DIR = Path(config.get("PATHS", "mff"))
SUBDAILY_MFF_DIR = SUBDAILY_MINGLE_DIR / config.get("PATHS", "mff_version")

# base dir to run on - GWS or scratch
ROOT_DIR = Path(config.get("PATHS", "root"))
DATESTAMP = config.get("PATHS", "version")


# set up suitable paths
# processing space - for intermediate files, between mff and qff
SUBDAILY_PROC_DIR = ROOT_DIR / config.get("PATHS", "proc") / DATESTAMP
if not SUBDAILY_PROC_DIR.exists():
    SUBDAILY_PROC_DIR.mkdir()

SUBDAILY_OUT_DIR = ROOT_DIR / config.get("PATHS", "qff") / DATESTAMP
if not SUBDAILY_OUT_DIR.exists():
    SUBDAILY_OUT_DIR.mkdir()
#    os.chmod(SUBDAILY_OUT_DIR, stat.S_IWGRP)

SUBDAILY_BAD_DIR = SUBDAILY_OUT_DIR / "bad_stations" #  datestamp in Subdaily_out_dir
if not SUBDAILY_BAD_DIR.exists():
    SUBDAILY_BAD_DIR.mkdir()

SUBDAILY_CONFIG_DIR = ROOT_DIR / config.get("PATHS", "config") / DATESTAMP
if not SUBDAILY_CONFIG_DIR.exists():
    SUBDAILY_CONFIG_DIR.mkdir()

SUBDAILY_IMAGE_DIR = ROOT_DIR / config.get("PATHS", "images") / DATESTAMP
if not SUBDAILY_IMAGE_DIR.exists():
    SUBDAILY_IMAGE_DIR.mkdir()

SUBDAILY_FLAG_DIR = ROOT_DIR / config.get("PATHS", "flags") / DATESTAMP
if not SUBDAILY_FLAG_DIR.exists():
    SUBDAILY_FLAG_DIR.mkdir()

SUBDAILY_LOG_DIR = ROOT_DIR / config.get("PATHS", "logs") / DATESTAMP
if not SUBDAILY_LOG_DIR.exists():
    SUBDAILY_LOG_DIR.exists()

SUBDAILY_ERROR_DIR = ROOT_DIR / config.get("PATHS", "errors") / DATESTAMP
if not SUBDAILY_ERROR_DIR.exists():
    SUBDAILY_ERROR_DIR.mkdir()

SUBDAILY_METADATA_DIR = ROOT_DIR / config.get("PATHS", "metadata") / DATESTAMP
if not SUBDAILY_METADATA_DIR.exists():
    SUBDAILY_METADATA_DIR.mkdir()

# name of station list
STATION_LIST = Path(config.get("FILES", "station_list"))
STATION_FULL_LIST = Path(config.get("FILES", "station_full_list"))
INVENTORY = Path(config.get("FILES", "inventory"))

# Check and set compression options
IN_COMPRESSION = config.get("FILES", "in_compression")
if IN_COMPRESSION == "None":
    IN_COMPRESSION = ""
OUT_COMPRESSION = config.get("FILES", "out_compression")
if OUT_COMPRESSION == "None":
    OUT_COMPRESSION = ""

# Check and set file format options
IN_FORMAT = config.get("FILES", "in_format")
if IN_FORMAT not in ("csv", "psv", "pqt", "parquet"):
    sys.exit("Error in `in_format` entry in config file")
OUT_FORMAT = config.get("FILES", "out_format")
if OUT_FORMAT not in ("csv", "psv", "pqt", "parquet"):
    sys.exit("Error in `out_format` entry in config file")

# Check and set file format options
IN_SUFFIX = config.get("FILES", "in_suffix")
if IN_SUFFIX not in (".mff", ".csv", ".psv", ".pqt", ".parquet"):
    sys.exit("Error in `in_suffix` entry in config file")
OUT_SUFFIX = config.get("FILES", "out_suffix")
if OUT_SUFFIX not in (".qff", ".csv", ".psv", ".pqt", ".parquet"):
    sys.exit("Error in `out_suffix` entry in config file")


#*********************************************
# read in parameter list
VARFILE = config.get("FILES", "variables")
with open(Path(__file__).parent / "configs" / VARFILE, "r") as pf:
    parameters = json.load(pf)
obs_var_list = parameters["variables"]["process_vars"]
carry_thru_var_list = parameters["variables"]["not_process_vars"]

DTYPE_DICT =  {}
for var_list in (obs_var_list, carry_thru_var_list):
    for v, var in enumerate(var_list):
        if var in ["STATION", "Station_name", "DATE",
                   "REM", "remarks",
                   "pressure_3hr_change", "sky_condition"]:
            DTYPE_DICT[var] = str
        elif "pres_wx" in var:
            # catch all present weather codes (which may contain character info)
            DTYPE_DICT[var] = str
        elif "baseht" in var:
            # catch the baseheight entries before the sky_cover ones, as these are floats
            DTYPE_DICT[var] = np.float64
        elif "sky_cover" in var:
            # catch all sky cover codes (which may contain character info)
            DTYPE_DICT[var] = str
        else:
            DTYPE_DICT[var] = np.float64

        DTYPE_DICT[f"{var}_Source_ID"] = str
        DTYPE_DICT[f"{var}_QC_flag"] = str
        DTYPE_DICT[f"{var}_Measurement_Code"] = str
        DTYPE_DICT[f"{var}_Quality_Code"] = str
        DTYPE_DICT[f"{var}_Report_Type"] = str
        DTYPE_DICT[f"{var}_Source_Code"] = str
        DTYPE_DICT[f"{var}_Source_Station_ID"] = str
#        DTYPE_DICT[f"Source_ID.{v+1}")] = str

DTYPE_DICT["Source_ID"] = str


# get the wind measurement codes
with open(Path(__file__).parent / "configs" / "wind_measurement_codes.json", 'r') as infile:
    WIND_MEASUREMENT_CODES = json.load(infile)



#************************************************************************
if __name__ == "__main__":

    # if called as stand alone, then clean out the errors from a previous run
    print("Removing errors from previous runs")
    if len(SUBDAILY_ERROR_DIR.iterdir()) != 0:
        for this_file in SUBDAILY_ERROR_DIR.iterdir():
            this_file.unlink()


#*********************************************
# END
#*********************************************
