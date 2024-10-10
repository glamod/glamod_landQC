"""
Contains tests for precision.py
"""
import os
import numpy as np
import datetime as dt
import pandas as pd
import pytest
from unittest.mock import patch, Mock

import io_utils
import qc_utils
import setup

import common


EXAMPLE_FILES = os.listdir(os.path.join(os.path.dirname(__file__),
                           "example_data"))


def test_read_psv() -> None:

    infile = os.path.join(os.path.dirname(__file__),
                          "example_data", EXAMPLE_FILES[0])
    separator = "|"

    df = io_utils.read_psv(infile, separator)

    assert len(df.columns) == 238+len(setup.obs_var_list)
    assert df.shape[0] == 144 # checked manually, and rows ignore header


def test_read_psv_fileerror() -> None:

    infile = os.path.join(os.path.dirname(__file__),
                          "example_data", EXAMPLE_FILES[0])
    separator = "|"

    # is a FileNotFoundError raised
    with pytest.raises(FileNotFoundError) as emsg:
        _ = io_utils.read_psv(infile+"f", separator)

    assert "No such file or directory:" in str(emsg)


@patch("io_utils.read_psv")
def test_read(read_psv_mock: Mock) -> None:

    infile = os.path.join(os.path.dirname(__file__),
                          "example_data", EXAMPLE_FILES[0])

    _ = io_utils.read(infile)

    read_psv_mock.assert_called_once_with(infile, "|")



def test_read_oserror() -> None:

    infile = os.path.join(os.path.dirname(__file__),
                          "example_data", EXAMPLE_FILES[0])

    # is a FileNotFoundError raised
    with pytest.raises(FileNotFoundError):
        _ = io_utils.read(infile+"f")


#def test_read_station() -> None:


#def test_write_psv() -> None:

def test_write(tmp_path) -> None:
        
    outfile = os.path.join(tmp_path, "dummy_file.psv")

    data = {"ID" : ["dummy", "dummy"],
            "Latitude" : [40.39, 40.39],
            "Longitude" : [30.49, 30.49],
            "temperatures" : [0, 10]}
    df = pd.DataFrame(data)

    io_utils.write(outfile, df)

    with open(os.path.join(tmp_path, "dummy_file.psv"), "r") as infile:
        written_frame = infile.readlines()

    assert written_frame[0] == "|".join([key for key, _ in data.items()]) + "\n"
    assert written_frame[1] == "|".join([f"{vals[0]}" for _, vals in data.items()]) + "\n"
    assert written_frame[2] == "|".join([f"{vals[1]}" for _, vals in data.items()]) + "\n"


def test_write_formatters(tmp_path) -> None:
        
    outfile = os.path.join(tmp_path, "dummy_file.psv")

    data = {"ID" : ["dummy", "dummy"],
            "Latitude" : [40.39, 40.39],
            "Longitude" : [30.49, 30.49],
            "temperatures" : [0, 10]}
    df = pd.DataFrame(data)

    io_utils.write(outfile, df, formatters={"Latitude" : "{:7.4f}", "Longitude" : "{:7.4f}"})

    with open(os.path.join(tmp_path, "dummy_file.psv"), "r") as infile:
        written_frame = infile.readlines()

    # in this case hard code the string to match
    assert written_frame[0] == "|".join([key for key, _ in data.items()]) + "\n"
    assert written_frame[1] == "dummy|40.3900|30.4900|0\n"


# def test_flag_write() -> None:
        
@patch("io_utils.setup")
def test_write_error(setup_mock: Mock,
                     tmp_path) -> None:
    
    station = qc_utils.Station("DMY01234567", 50, 50, 50)
    setup_mock.DATESTAMP = "DUMMYDATE"
    setup_mock.SUBDAILY_ERROR_DIR = tmp_path

    io_utils.write_error(station, "test message")

    with open(os.path.join(tmp_path, "DMY01234567.err"), "r") as infile:
        written_message = infile.readlines()

    assert written_message[-1] == "test message\n"


@patch("io_utils.setup")
def test_write_error_append(setup_mock: Mock,
                     tmp_path) -> None:
    
    station = qc_utils.Station("DMY01234567", 50, 50, 50)
    setup_mock.SUBDAILY_ERROR_DIR = tmp_path

    # create the file and check append step worked
    with open(os.path.join(tmp_path, "DMY01234567.err"), "w") as outfile:
        outfile.write("Existing error message\n")

    io_utils.write_error(station, "test message")

    with open(os.path.join(tmp_path, "DMY01234567.err"), "r") as infile:
        written_message = infile.readlines()

    assert written_message[0] == "Existing error message\n"