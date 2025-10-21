"""
Contains tests for convert_to_yearly_parquet.py
"""
from pathlib import Path
import pandas as pd
from unittest.mock import patch, Mock
import pytest

import convert_to_yearly_parquet

EXAMPLE_FILES = [f for f in (Path(__file__).parent / "example_data").glob("*.qff")]

@pytest.mark.parametrize("compression", ("", ".gz", ".zip"))
@patch("convert_to_yearly_parquet.setup")
@patch("convert_to_yearly_parquet.Path.iterdir")
def test_get_files(iterdir_mock: Mock,
                   setup_mock: Mock,
                   compression: str) -> None:


    iterdir_mock.return_value = [Path("test1.qff"),
                                 Path("test2.qff"),
                                 Path("test3.qff.gz"),
                                 Path("test4.qff.gz"),
                                 Path("test5.qff.zip"),
                                 Path("test6.qff.zip")]
    setup_mock.OUT_COMPRESSION = compression
    setup_mock.SUBDAILY_OUT_DIR = Path("testdir")

    qff_files = convert_to_yearly_parquet.get_files()

    # do manually
    if compression == "":
        assert qff_files == [Path("test1.qff"), Path("test2.qff")]
    elif compression == ".gz":
        assert qff_files == [Path("test3.qff.gz"), Path("test4.qff.gz")]
    elif compression == ".zip":
        assert qff_files == [Path("test5.qff.zip"), Path("test6.qff.zip")]


@patch("convert_to_yearly_parquet.setup")
def test_process_files(setup_mock: Mock) -> None:

    # Only testing unzipped files, so that these are more easily checked
    setup_mock.OUT_COMPRESSION = ""
    setup_mock.SUBDAILY_OUT_DIR = Path(__file__).parent / "example_data"

    yearly_data = convert_to_yearly_parquet.process_files(EXAMPLE_FILES)

    # check keys are correct
    for key in yearly_data.keys():
        assert key in [1979, 1985]

    # check length of Dataframes for each key are correct
    for frame in yearly_data[1979]:
        # can't presume the order will always be the same
        assert frame.shape[0] in (40, 144, 150, 358, 496)
    assert yearly_data[1985][0].shape[0] == 40


@patch("convert_to_yearly_parquet.pd")
@patch("convert_to_yearly_parquet.setup")
def test_process_files_error(setup_mock: Mock,
                             pd_mock: Mock) -> None:

    # Only testing unzipped files, so that these are more easily checked
    setup_mock.SUBDAILY_OUT_DIR = Path(__file__).parent / "example_data"

    # Create a data frame which will trigger the error
    erroneous_df = pd.read_csv(Path(__file__).parent / "example_data" / EXAMPLE_FILES[0],
                                        dtype=str, sep="|")
    del erroneous_df["Year"]

    pd_mock.read_csv.return_value = erroneous_df

    with pytest.raises(RuntimeError) as emsg:
        _ = convert_to_yearly_parquet.process_files(EXAMPLE_FILES)

    assert "Column 'Year' not found in" in str(emsg)




@patch("convert_to_yearly_parquet.setup")
def test_write_pqt(setup_mock: Mock,
                   tmp_path) -> None:

    # Only testing unzipped files, so that these are more easily checked
    setup_mock.DATESTAMP = "DUMMYDATE"
    setup_mock.SUBDAILY_OUT_DIR = Path(__file__).parent / "example_data"

    setup_mock.ROOT_DIR = tmp_path
    expected_outlocation = tmp_path / "pqt" / setup_mock.DATESTAMP

    yearly_data = convert_to_yearly_parquet.process_files(EXAMPLE_FILES)
    convert_to_yearly_parquet.write_pqt(yearly_data)

    # check correct number of files written
    assert len(list(expected_outlocation.iterdir())) == 2
    # with the correct names
    for outfile in ["qff_1979.parquet", "qff_1985.parquet"]:
        assert expected_outlocation / outfile in list(expected_outlocation.iterdir())

    # and when read in, they are the expected shape
    written_df = pd.read_parquet(expected_outlocation / "qff_1985.parquet",
                                 engine="pyarrow")

    assert written_df.shape == (40, 335)


@patch("convert_to_yearly_parquet.write_pqt")
@patch("convert_to_yearly_parquet.process_files")
@patch("convert_to_yearly_parquet.get_files")
def test_main(get_files_mock: Mock,
              process_files_mock: Mock,
              write_pqt_mock: Mock) -> None:

    convert_to_yearly_parquet.main()

    # check all the calls are made as expected
    get_files_mock.assert_called_once()
    process_files_mock.assert_called_once()
    write_pqt_mock.assert_called_once()

