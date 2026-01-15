#!/usr/bin/env python
'''
utils.py contains utility scripts to help with overall flow of suite
'''
from pathlib import Path
import configparser
import json
import pandas as pd
import numpy as np
import pathlib
import logging

import setup


UNIT_DICT = {"temperature" : "degrees C", \
             "dew_point_temperature" :  "degrees C", \
             "wind_direction" :  "degrees", \
             "wind_speed" : "meters per second", \
             "sea_level_pressure" : "hPa hectopascals", \
             "station_level_pressure" : "hPa hectopascals"}

# Lowercase letters for flags which should exclude data
# No information flags (the data are valid, but not necessarily adhering to conventions)
QC_TESTS = {"a" : "Repeated Day streaks",  # repeAted day streaks
            "b" : "Distribution - all",  # distriBution (all)
            "c" : "Climatological",  # Climatological
            "d" : "Distribution - monthly",  # Distribution (monthly)
            "e" : "Clean Up",  # clEan up
            "f" : "Frequent Value",  # Frequent value
            "h" : "High Flag Rate",  # High flag rate
            "i" : "Precision",  # precIsion
            "k" : "Repeating Streaks",  # repeating streaKs
            "l" : "Logic",  # Logic
            "m" : "Humidity",  # huMidity
            "n" : "Neighbour",  # Neighbour
            "o" : "Odd Cluster",  # Odd cluster
            "p" : "Pressure",  # Pressure
            "r" : "World Records",  # world Records
            "s" : "Spike",  # Spike
            "t" : "Timestamp",  # Timestamp
            "u" : "Diurnal",  # diUrnal
            "v" : "Variance",  # Variance
            "w" : "Winds",  # Winds
            "x" : "Excess streak proportion",  # eXcess streak proportion
            "z" : "Wind logical - calm, masked zero direction",
#            "," : "Timestamp - identical observation values",
            }


MDI = -1.e30
FIRST_YEAR = 1700
# Can have these values for station elevations which are allowed, but indicate missing
#   information.  Blank entries also allowed, but others which do not make sense
#   are caught by logic check.  Need to escape these for e.g. pressure checks
ALLOWED_MISSING_ELEVATIONS = ["-999", "9999"]

# These data are retained and processed by the QC tests.  All others are not.
WIND_MEASUREMENT_CODES = ["", "N-Normal", "C-Calm", "V-Variable", "9-Missing"]


#*********************************************
# Process the Configuration File
#*********************************************

CONFIG_FILE = Path(__file__).parent / "configuration.txt"

if not CONFIG_FILE.exists:
    print(f"Configuration file missing - {CONFIG_FILE}")
    quit()

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

#*********************************************
# Statistics
MEAN = config.getboolean("STATISTICS", "mean")
MEDIAN = config.getboolean("STATISTICS", "median")
if MEAN == MEDIAN:
    print("Configuration file STATISTICS entry malformed. One of mean or median only")
    quit()


# MAD = 0.8 SD
# IQR = 1.3 SD
STDEV = config.getboolean("STATISTICS", "stdev")
IQR = config.getboolean("STATISTICS", "iqr")
MAD = config.getboolean("STATISTICS", "mad")
if sum([STDEV, MAD, IQR]) >= 2:
    print("Configuration file STATISTICS entry malformed. One of stdev, iqr, median only")
    quit()

#*********************************************
# Thresholds
DATA_COUNT_THRESHOLD = config.getint("THRESHOLDS", "min_data_count")
HIGH_FLAGGING = config.getfloat("THRESHOLDS", "high_flag_proportion")
ODD_CLUSTER_SEPARATION = config.getint("THRESHOLDS", "odd_cluster_separation")

# read in logic check list
LOGICFILE = Path(__file__).parent / "configs" / config.get("FILES", "logic")

#*********************************************
# Neighbour Checks
MAX_NEIGHBOUR_DISTANCE = config.getint("NEIGHBOURS", "max_distance")
MAX_NEIGHBOUR_VERTICAL_SEP = config.getint("NEIGHBOURS", "max_vertical_separation")
MAX_N_NEIGHBOURS = config.getint("NEIGHBOURS", "max_number")
NEIGHBOUR_FILE = setup.SUBDAILY_CONFIG_DIR / config.get("NEIGHBOURS", "filename")
MIN_NEIGHBOURS = config.getint("NEIGHBOURS", "minimum_number")

#*********************************************
# Set up the Classes
#*********************************************
class MeteorologicalVariable(object):
    '''
    Class for meteorological variable.  Initialised with metadata only
    '''

    def __init__(self, name: str, mdi: float,
                 units: str, dtype: str):

        self.name = name
        self.mdi = mdi
        self.units = units
        self.dtype = dtype


    def __str__(self):
        return f"variable: {self.name}"


    __repr__ = __str__


    def store_data(self, indata: np.ma.MaskedArray):
        """Set data array for observations"""
        self.data: np.ma.MaskedArray = indata


    def store_flags(self, flags: np.ndarray):
        """Store the flag information"""
        self.flags: np.ndarray = flags



#*********************************************
class Station(object):
    '''
    Class for station
    '''

    def __init__(self, stn_id: str,
                 lat: float, lon: float,
                 elev: float):
        self.id = stn_id
        self.lat = lat
        self.lon = lon
        self.elev = elev

        # set other information
        self.country: str = ""
        self.continent: str = ""

        # set up empty placeholders of observation data
        for obs_var in setup.obs_var_list:
            setattr(self, obs_var, None)


    def __str__(self):
        return f"station {self.id}, lat {self.lat}, lon {self.lon}, elevation {self.elev}"


    __repr__ = __str__


    def set_times(self, times: pd.Series):
        """Set the times attribute"""
        self.times: pd.Series = times


    def set_datetime_values(self, years: np.ndarray,
                            months: np.ndarray,
                            days: np.ndarray,
                            hours: np.ndarray):
        """Set values (arrays) for each date quantity"""

        assert years.shape == months.shape == days.shape == hours.shape

        self.years: np.ndarray = years
        self.months: np.ndarray = months
        self.days: np.ndarray = days
        self.hours: np.ndarray = hours



#************************************************************************
# Subroutines
#************************************************************************
def get_station_list(restart_id: str = "", end_id: str = "") -> pd.DataFrame:
    """
    Read in station list file(s) and return dataframe

    :param str restart_id: which station to start on
    :param str end_id: which station to end on

    :returns: dataframe of station list
    """
    # Test if station list fixed-width format or comma separated
    #  [Initially supplied nonFWF format for Release 8 processing]
    fwf = True
    with open(setup.STATION_LIST, "r") as infile:
        lines = infile.readlines()
        for row in lines:
            # The non FWF version had double quotes present around the station names
            #    but was space separated, so can use that to determine if it's FWF or not
            if row.find('"') != -1:
                fwf = False

    # process the station list
    if fwf:
        # Fixed width format
        # If station-ID has format "ID-START-END" then width is 29
        station_list = pd.read_fwf(setup.STATION_LIST, widths=(11, 9, 10, 7, 3, 40, 5),
                                header=None, names=("id", "latitude", "longitude", "elevation", "state",
                                                    "name", "wmo"))
    else:
        # Comma separated
        station_list = pd.read_csv(setup.STATION_LIST, delim_whitespace=True,
                                header=None, names=("id", "latitude", "longitude", "elevation", "name"))
        # add extra columns (despite being empty) so these are available to later stages
        #  use insert for "state" so that order of columns is the same
        station_list.insert(4, "state", ["" for i in range(len(station_list))])
        station_list["wmo"] = ["" for i in range(len(station_list))]

    # fill empty entries (default NaN) with blank strings
    station_list = station_list.fillna("")

    station_IDs = station_list.id

    # work from the end to save messing up the start indexing
    if end_id != "":
        endindex, = station_IDs.index[station_IDs == end_id].to_list()
        station_list = station_list.iloc[: endindex+1]

    # and do the front
    if restart_id != "":
        startindex, = station_IDs.index[station_IDs == restart_id].to_list()
        station_list = station_list.iloc[startindex:]

    return station_list.reset_index(drop=True) # get_station_list


#************************************************************************
def insert_flags(qc_flags: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """
    Update QC flags with the new flags

    :param array qc_flags: string array of flags
    :param array flags: string array of flags
    """

    qc_flags = np.char.add(qc_flags.astype(str), flags.astype(str))

    return qc_flags # insert_flags


#************************************************************************
def get_measurement_code_mask(ds: pd.Series,
                              measurement_codes: list) -> pd.Series:
    """
    Build up a mask of data rows to ignore by using a list of permitted
    measurement codes

    Parameters
    ----------
    ds : pd.Series
        Measurement Codes field from data frame for variable
    measurement_codes : list
        List of accepted codes

    Returns
    -------
    pd.Series
        Boolean array of mask
    """
    assert isinstance(ds, pd.Series)
    assert isinstance(measurement_codes, list)

    # Build up the mask
    for c, code in enumerate(measurement_codes):

        if code == "":
            # Empty flags converted to NaNs on reading
            if c == 0:
                mask = (ds.isna())
            else:
                mask = (ds.isna()) | mask
        else:
            # Doing string comparison, but need to exclude NaNs
            #   Need to convert to string before assessing (np.nan -> "nan") [using .astype(str)]
            #   But test for NaNs separately [using .isna()], so that a string starting "nan"
            #   could be used in the future
            if c == 0:
                # Initialise
                mask = (~ds.isna() & ds.astype(str).str.startswith(code))
            else:
                # Combine using Or symbol ("|")
                #   e.g. if code = "N-Normal" or "C-Calm" or "" set True
                mask = (~ds.isna() & ds.astype(str).str.startswith(code)) | mask

    return mask


#************************************************************************
def populate_station(station: Station, df: pd.DataFrame, obs_var_list: list, read_flags: bool = False) -> None:
    """
    Convert Data Frame into internal station and obs_variable objects

    :param Station station: station object to hold information
    :param DataFrame df: dataframe of input data
    :param list obs_var_list: list of observed variables
    :param bool read_flags: read in already pre-existing flags
    """

    for variable in obs_var_list:

        # make a variable
        this_var = MeteorologicalVariable(variable, MDI, UNIT_DICT[variable],
                                          "float")

        # store the data
        indata = df[variable].fillna(MDI).to_numpy()
        indata = indata.astype(float)

        # For wind direction and speed only, account for some measurement flags
        #  Mask data in the Met_Var object used for the tests, but leave dataframe
        #  unaffected.
        if variable in ["wind_direction", "wind_speed"]:
            m_code = df[f"{variable}_Measurement_Code"]
            measurement_codes = setup.WIND_MEASUREMENT_CODES[variable]["retained"]

            mask = get_measurement_code_mask(m_code, measurement_codes)

            # invert mask and set to missing
            indata[~mask] = MDI

        this_var.store_data(np.ma.masked_where(indata == MDI, indata))

        if len(this_var.data.mask.shape) == 0:
            # single mask value, replace with arrage of True/False's
            if this_var.data.mask:
                # True
                this_var.data.mask = np.ones(this_var.data.shape)
            else:
                # False
                this_var.data.mask = np.zeros(this_var.data.shape)

        this_var.data.fill_value = MDI

        if read_flags:
            # change all empty values (else NaN) to blank
            this_var.store_flags(df[f"{variable}_QC_flag"].fillna("").to_numpy())
        else:
            # empty flag array
            this_var.store_flags(np.array(["" for i in range(len(this_var.data))]))

        # and store
        setattr(station, variable, this_var)

    # populate_station


#************************************************************************
def find_country_code(lat: float, lon: float) -> str:
    """
    Use reverse Geocoder to find closest city to each station, and hence
    find the country code.

    :param float lat: latitude
    :param float lon: longitude

    :returns: [str] country_code
    """
    import reverse_geocoder as rg
    results = rg.search((lat, lon))
    country = results[0]['cc']

    return country # find_country_code

#************************************************************************
def find_continent(country_code: str) -> str:
    """
    Use ISO country list to find continent from country_code.

    :param str country_code: ISO standard country code

    :returns: [str] continent
    """

    # as maybe run from another directory, get the right path
    cwd = pathlib.Path(__file__).parent.absolute()
    # prepare look up
    with open(f'{cwd}/configs/iso_country_codes.json', 'r') as infile:
        iso_codes = json.load(infile)

    concord = {}
    for entry in iso_codes:
        concord[entry["Code"]] = entry["continent"]

    return concord[country_code]


#************************************************************************
def custom_logger(logfile: Path):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Remove any current handlers; the handlers persist within the same
    # Python session, so ensure the root logger is 'clean' every time
    # this function is called. Note that using logger.removeHandler()
    # doesn't work reliably
    logger.handlers = []

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    # create file handler to capture all output
    fh = logging.FileHandler(logfile, "w")
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    logconsole_format = logging.Formatter('%(levelname)-8s %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(logconsole_format)

    logfile_format = logging.Formatter('%(asctime)s %(module)s %(levelname)-8s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(logfile_format)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
