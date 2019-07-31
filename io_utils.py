'''
io_utils - contains scripts for read/write of main files
'''


import os
import pandas as pd
import numpy as np
import datetime as dt


#************************************************************************
def read_psv(infile, separator, compression="infer"):
    '''

    http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table
    '''
    df = pd.read_csv("{}.psv".format(infile), sep=separator, compression=compression)

    return df #  read_psv


#************************************************************************
def read(infile):
    """
    Wrapper for read functions to allow remainder to be file format agnostic.

    :param str infile: location and name of infile (without extension)
    
    :returns: df - DataFrame
    """

    # for .psv
    df = read_psv(infile, "|")

    return df # read

#************************************************************************
def write_psv(outfile, df, separator, compression="infer"):
    '''

    http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table
    '''
    df.to_csv("{}.psv".format(outfile), sep=separator, compression=compression)

    return # write_psv

#************************************************************************
def write(outfile, df):
    """
    Wrapper for write functions to allow remainder to be file format agnostic.

    :param str outfile: location and name of outfile (without extension)
    :param DataFrame df: data frame to write
    """

    # for .psv
    write_psv(outfile, df, "|")

    return # write
