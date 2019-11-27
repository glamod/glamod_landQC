'''
initial_setup - contains settings and path information
'''


#*********************************************
import os
import configparser
import sys
import json


#*********************************************
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
ROOT_DIR = config.get("PATHS", "root")
SUBDAILY_IN_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "mff"), config.get("PATHS", "mff_version"))
SUBDAILY_ROOT_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "mff"))

SUBDAILY_OUT_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "qff"), config.get("PATHS", "qff_version"))
if not os.path.exists(SUBDAILY_OUT_DIR):
    os.mkdir(SUBDAILY_OUT_DIR)

SUBDAILY_BAD_DIR = os.path.join(SUBDAILY_OUT_DIR, "bad_stations")
if not os.path.exists(SUBDAILY_BAD_DIR):
    os.mkdir(SUBDAILY_BAD_DIR)

SUBDAILY_CONFIG_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "config"))
if not os.path.exists(SUBDAILY_CONFIG_DIR):
    os.mkdir(SUBDAILY_CONFIG_DIR)

SUBDAILY_IMAGE_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "images"))
if not os.path.exists(SUBDAILY_IMAGE_DIR):
    os.mkdir(SUBDAILY_IMAGE_DIR)

SUBDAILY_ERROR_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "errors"))
if not os.path.exists(SUBDAILY_ERROR_DIR):
    os.mkdir(SUBDAILY_ERROR_DIR)

# for cross-timescale checks
# DAILY_DIR = 
# MONTHLY_DIR = 

#*********************************************
# run type
# TODO - is it best to do this here or with a command line switch?
FULL_UPDATE = config.getboolean("SETTINGS", "full_reprocess")
# if False, then it's a NRT append - reading from the defined files if set


#*********************************************
# read in parameter list
VARFILE = config.get("FILES", "variables")
with open(os.path.join(os.path.dirname(__file__), VARFILE), "r") as pf:
    parameters = json.load(pf)
obs_var_list = parameters["variables"]["process_vars"]

DTYPE_DICT =  {}
for var in obs_var_list:
    DTYPE_DICT[var] = float
    DTYPE_DICT["{}_Source_ID".format(var)] = int
    DTYPE_DICT["{}_QC_flag".format(var)] = str


#************************************************************************
if __name__ == "__main__":

    # if called as stand alone, then clean out the errors from a previous run
    if len(os.listdir(SUBDAILY_ERROR_DIR)) != 0:
        for this_file in os.listdir(SUBDAILY_ERROR_DIR):
            os.remove(os.path.join(SUBDAILY_ERROR_DIR, this_file))
        

#*********************************************
# END
#*********************************************
