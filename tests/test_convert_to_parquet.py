"""
Contains tests for convert_to_parquet.py
"""
import os
import glob
import pandas as pd
from unittest.mock import patch, Mock
import pytest

import convert_to_parquet

EXAMPLE_FILES = glob.glob(os.path.join(os.path.dirname(__file__),
                           "example_data", "*.?ff"))

@pytest.mark.parametrize("compression", ("", ".gz", ".zip"))
@patch("convert_to_parquet.setup")
@patch("convert_to_parquet.os")
def test_get_files(os_mock: Mock,
                   setup_mock: Mock,
                   compression: str) -> None:


    os_mock.listdir.return_value = ["test1.qff",
                                    "test2.qff",
                                    "test3.qff.gz",
                                    "test4.qff.gz",
                                    "test5.qff.zip",
                                    "test6.qff.zip"]
    setup_mock.OUT_COMPRESSION = compression
    setup_mock.SUBDAILY_OUT_DIR = "testdir/"

    qff_files = convert_to_parquet.get_files()

    # do manually
    if compression == "":
        assert qff_files == ["test1.qff", "test2.qff"]
    elif compression == ".gz":
        assert qff_files == ["test3.qff.gz", "test4.qff.gz"]
    elif compression == ".zip":
        assert qff_files == ["test5.qff.zip", "test6.qff.zip"]    
   

@patch("convert_to_parquet.setup")
def test_process_files(setup_mock: Mock) -> None:

    # Only testing unzipped files, so that these are more easily checked
    setup_mock.OUT_COMPRESSION = ""
    setup_mock.SUBDAILY_OUT_DIR = os.path.join(os.path.dirname(__file__),
                                               "example_data")

    yearly_data = convert_to_parquet.process_files(EXAMPLE_FILES)

    # check keys are correct
    for key in yearly_data.keys():
        assert key in [1979, 1985]

    # check length of Dataframes for each key are correct
    for frame in yearly_data[1979]:
        # can't presume the order will always be the same
        assert frame.shape[0] in (144, 358)
    assert yearly_data[1985][0].shape[0] == 40


@patch("convert_to_parquet.pd")
@patch("convert_to_parquet.setup")
def test_process_files_error(setup_mock: Mock,
                             pd_mock: Mock) -> None:

    # Only testing unzipped files, so that these are more easily checked
    setup_mock.SUBDAILY_OUT_DIR = os.path.join(os.path.dirname(__file__),
                                               "example_data")
    
    # Create a data frame which will trigger the error
    erroneous_df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                        "example_data", EXAMPLE_FILES[0]),
                                        dtype=str, sep="|")
    del erroneous_df["Year"]

    pd_mock.read_csv.return_value = erroneous_df

    with pytest.raises(RuntimeError) as emsg:
        _ = convert_to_parquet.process_files(EXAMPLE_FILES)

    assert "Column 'Year' not found in" in str(emsg)

  


@patch("convert_to_parquet.setup")
def test_write_pqt(setup_mock: Mock,
                   tmp_path) -> None:

    # Only testing unzipped files, so that these are more easily checked
    setup_mock.DATESTAMP = "DUMMYDATE"
    setup_mock.SUBDAILY_OUT_DIR = os.path.join(os.path.dirname(__file__),
                                               "example_data")

    setup_mock.ROOT_DIR = tmp_path
    expected_outlocation = os.path.join(tmp_path, "pqt", setup_mock.DATESTAMP)

    yearly_data = convert_to_parquet.process_files(EXAMPLE_FILES)
    convert_to_parquet.write_pqt(yearly_data)

    # check correct number of files written
    assert len(os.listdir(expected_outlocation)) == 2
    # with the correct names
    for outfile in ["qff_1979.parquet", "qff_1985.parquet"]:
        assert outfile in os.listdir(expected_outlocation)

    # and when read in, they are the expected shape
    written_df = pd.read_parquet(os.path.join(expected_outlocation,
                                              "qff_1985.parquet"),
                                 engine="pyarrow")
    
    assert written_df.shape == (40, 244)

    
@patch("convert_to_parquet.write_pqt")
@patch("convert_to_parquet.process_files")
@patch("convert_to_parquet.get_files")
def test_main(get_files_mock: Mock,
              process_files_mock: Mock,
              write_pqt_mock: Mock) -> None:
    
    convert_to_parquet.main()

    # check all the calls are made as expected
    get_files_mock.assert_called_once()
    process_files_mock.assert_called_once()
    write_pqt_mock.assert_called_once()

