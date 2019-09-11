'''
io_utils - contains scripts for read/write of main files
'''

import pandas as pd

#************************************************************************
def read_psv(infile, separator, extension="mff", compression="infer"):
    '''

    http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table

    :param str infile: location and name of infile (without extension)
    :param str separator: separating character (e.g. ",", "|")
    :param str extension: infile extension [mff]
    :returns: df - DataFrame

    '''
    df = pd.read_csv("{}.{}".format(infile, extension), sep=separator, compression=compression)

    return df #  read_psv

#************************************************************************
def read(infile, extension="mff"):
    """
    Wrapper for read functions to allow remainder to be file format agnostic.

    :param str infile: location and name of infile (without extension)
    :param str extension: infile extension [mff]
    :returns: df - DataFrame
    """

    # for .psv
    df = read_psv(infile, "|", extension=extension)

    return df # read

#************************************************************************
def write_psv(outfile, df, separator, extension="qff", compression="infer"):
    '''
    http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table

    :param str outfile: location and name of outfile (without extension)
    :param DataFrame df: data frame to write
    :param str separator: separating character (e.g. ",", "|")
    :param str extension: infile extension [qff]
    '''
    df.to_csv("{}.{}".format(outfile, extension), index=False, sep=separator, compression=compression)

    return # write_psv

#************************************************************************
def write(outfile, df, extension="qff"):
    """
    Wrapper for write functions to allow remainder to be file format agnostic.

    :param str outfile: location and name of outfile (without extension)
    :param str extension: infile extension [qff]
    :param DataFrame df: data frame to write
    """

    # for .psv
    write_psv(outfile, df, "|", extension=extension)

    return # write
