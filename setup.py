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
SUBDAILY_IN_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "mff"))
SUBDAILY_OUT_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "qff"))
SUBDAILY_CONFIG_DIR = os.path.join(ROOT_DIR, config.get("PATHS", "config"))

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

#*********************************************
# END
#*********************************************
