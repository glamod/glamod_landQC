"""
Contains tests for precision.py
"""
import os
import numpy as np
import datetime as dt
import pandas as pd
from subprocess import CalledProcessError, CompletedProcess
import pytest
from unittest.mock import call, patch, Mock

import io_utils
import qc_utils
import setup


EXAMPLE_FILES = os.listdir(os.path.join(os.path.dirname(__file__),
                           "example_data"))

def test_count_skip_rows() -> None:

    infile = os.path.join(os.path.dirname(__file__),
                          "example_data", "AAI0000TNCA.mff")

    result = io_utils.count_skip_rows(infile)

    assert result[0] == 1000


def test_read_psv_mff() -> None:

    infile = os.path.join(os.path.dirname(__file__),
                          "example_data", "AAI0000TNCA.mff")
    separator = "|"

    df = io_utils.read_psv(infile, separator)

    assert len(df.columns) == 329
    assert df.shape[0] == 3107 # checked manually, and rows ignore header lines


def test_read_psv_qff() -> None:

    infile = os.path.join(os.path.dirname(__file__),
                          "example_data", "AJM00037843.qff")
    separator = "|"

    df = io_utils.read_psv(infile, separator)

    assert len(df.columns) == 329+len(setup.obs_var_list)
    assert df.shape[0] == 496 # checked manually, and rows ignore header


def test_read_psv_qff_fileerror() -> None:

    infile = os.path.join(os.path.dirname(__file__),
                          "example_data", "AJM00037843.qff")
    separator = "|"

    # is a FileNotFoundError raised
    with pytest.raises(FileNotFoundError) as emsg:
        _ = io_utils.read_psv(infile+"f", separator)

    assert "No such file or directory:" in str(emsg)


@patch("io_utils.read_psv")
@patch("io_utils.setup")
def test_read_with_psv(setup_mock: Mock,
                       read_psv_mock: Mock) -> None:

    setup_mock.IN_FORMAT = "psv"
    infile = os.path.join(os.path.dirname(__file__),
                          "example_data", "AJM00037843.qff")

    read_psv_mock.return_value = pd.DataFrame([["AJM00037843", "SABIRABAD", 1979],
                                               ["AJM00037843", "SABIRABAD", 1979]],
                                              columns=["STATION", "Station_name", "Year"])

    _ = io_utils.read(infile)

    read_psv_mock.assert_called_once_with(infile, "|")


def test_read_oserror() -> None:

    infile = os.path.join(os.path.dirname(__file__),
                          "example_data", "AJM00037843.qff")

    # is a FileNotFoundError raised
    with pytest.raises(FileNotFoundError):
        _ = io_utils.read(infile+"f")


def test_calculate_datetimes() -> None:

    data = {"Year" : [2020, 2024],
            "Month" : [1, 2],
            "Day" : [3, 4],
            "Hour" : [5, 6],
            "Minute" : [7, 8]}

    df = pd.DataFrame(data)

    datetimes = io_utils.calculate_datetimes(df)

    assert datetimes[0] == dt.datetime(2020, 1, 3, 5, 7)
    assert datetimes[1] == dt.datetime(2024, 2, 4, 6, 8)


def test_calculate_datetimes_error() -> None:

    data = {"Year" : [2020, 2024],
            "Month" : [1, 2],
            "Day" : [3, 30],
            "Hour" : [5, 6],
            "Minute" : [7, 8]}

    df = pd.DataFrame(data)

    with pytest.raises(ValueError) as emsg:
        _ = io_utils.calculate_datetimes(df)

    assert "Bad date - 2024-2-30" in str(emsg)


def test_convert_wind_flags() -> None:

    data = {"Year" : [2020, 2021, 2022, 2023, 2024],
            "Month" : [1, 2, 3, 4, 5],
            "Day" : [10, 11, 12, 13, 14],
            "wind_direction_Measurement_Code" : ["C-Calm", "V-Variable", "C-Calm", "Dummy", ""],
            "wind_direction" : [999, 999, 0, 90, 180]}

    df = pd.DataFrame(data)

    # set directions to NaN if C-Calm or V-Variable _and_ value = 999
    expected_data = {"Year" : [2020, 2021, 2022, 2023, 2024],
            "Month" : [1, 2, 3, 4, 5],
            "Day" : [10, 11, 12, 13, 14],
            "wind_direction_Measurement_Code" : ["C-Calm", "V-Variable", "C-Calm", "Dummy", ""],
            "wind_direction" : [np.nan, np.nan, 0, 90, 180]}
    expected_df = pd.DataFrame(expected_data)

    io_utils.convert_wind_flags(df)

    pd.testing.assert_frame_equal(df, expected_df)



def test_read_station() -> None:

    # Not ideal, but as setup is used to determine other
    #   aspects, mocking this ended up with recursion errors
    if setup.IN_FORMAT in ["psv", "csv"]:
        infile = os.path.join(os.path.dirname(__file__),
                            "example_data", "AJM00037898.qff")
    elif setup.IN_FORMAT in ["pqt", "parquet"]:
        infile = os.path.join(os.path.dirname(__file__),
                            "example_data", "AJM00037898.pqt")

    station = qc_utils.Station("AJM00037898", 39.6500, 46.5330, 1099.0)

    station, station_df = io_utils.read_station(infile, station)

    # use example data, so these values will always be right
    assert station.years[0] == 1979
    assert station.months[0] == 8
    assert station.days[0] == 14
    assert station.hours[0] == 0
    assert station.times[0] == dt.datetime(1979, 8, 14, 0, 0)

    assert station_df.shape == (144, 329+len(setup.obs_var_list))


def test_read_station_error() -> None:

    # unreachable file
    infile = os.path.join(os.path.dirname(__file__),
                          "example_data", "AJM000DUMMY.mff")

    station = qc_utils.Station("AJM000DUMMY", 39.6500, 46.5330, 1099.0)

    with pytest.raises(FileNotFoundError):
        station, _ = io_utils.read_station(infile, station)


def test_write_psv(tmp_path) -> None:
    separator   = "|"
    outfile = os.path.join(tmp_path, "dummy_file.psv")

    data = {"ID" : ["dummy", "dummy"],
            "Latitude" : [40.39, 40.39],
            "Longitude" : [30.49, 30.49],
            "temperatures" : [0, 10]}
    df = pd.DataFrame(data)

    io_utils.write_psv(outfile, df, separator)

    with open(os.path.join(tmp_path, "dummy_file.psv"), "r") as infile:
        written_frame = infile.readlines()

    assert written_frame[0] == "|".join([key for key, _ in data.items()]) + "\n"
    assert written_frame[1] == "|".join([f"{vals[0]}" for _, vals in data.items()]) + "\n"
    assert written_frame[2] == "|".join([f"{vals[1]}" for _, vals in data.items()]) + "\n"


@patch("io_utils.logging")
@patch("io_utils.subprocess.run")
def test_integrity_check_fail(run_mock: Mock,
                              logging_mock: Mock) -> None:

    run_mock.side_effect = CalledProcessError(returncode=1,
                                                         cmd="gzip -t dummy_file.psv",
                                                         stderr="error text")

    result = io_utils.integrity_check("dummy_file.psv")

    assert not result  # (i.e. False)
    calls = [call("Validation failed on dummy_file.psv"),
             call("  gzip -t dummy_file.psv"),
             call("  error text")]
    logging_mock.info.assert_has_calls(calls)


@patch("io_utils.logging")
@patch("io_utils.subprocess.run")
def test_integrity_check_pass(run_mock: Mock,
                              logging_mock: Mock) -> None:

    run_mock.return_value(CompletedProcess(["gzip", "-t", "dummy_file.psv"],
                                                      returncode=0,
                                                      stdout="output",
                                                      stderr=None))

    result = io_utils.integrity_check("dummy_file.psv")

    assert result  # (i.e. True)


@patch("io_utils.setup")
def test_write_with_psv(setup_mock: Mock,
                        tmp_path) -> None:

    setup_mock.OUT_FORMAT = "psv"
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


@patch("io_utils.setup")
def test_write_formatters_with_psv(setup_mock: Mock,
                                   tmp_path) -> None:

    setup_mock.OUT_FORMAT = "psv"
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


@patch("io_utils.setup")
def test_flag_write(setup_mock: Mock,
                    tmp_path) -> None:

    outfilename = os.path.join(tmp_path, "DMY01234567.flg")
    setup_mock.obs_var_list = ["temperature"]

    data = {"temperature" : [1, 2, 3, 4, 5],
            "temperature_QC_flag" : ["CFE", "CE", "Cw", "w", ""]}
    df = pd.DataFrame(data)

    io_utils.flag_write(outfilename, df)

    with open(os.path.join(tmp_path, "DMY01234567.flg"), "r") as infile:
        written_message = infile.readlines()

    with open(os.path.join(os.path.dirname(__file__),
                           "example_data",
                           "Example_flag_file.flg"), "r") as infile:
        expected_message = infile.readlines()

    np.testing.assert_array_equal(written_message, expected_message)


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
