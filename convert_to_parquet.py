'''
Convert station QFF files (psv format) to parquet format

convert_to_parquet.py invoked by typing::

  python convert_to_parquet.py [--diagnostics] [--clobber]

Input arguments:

--diagnostics       [False] Verbose output

--clobber           Overwrite output files if already existing.  If not set, will skip if output exists
'''
#************************************************************************
import os
import pandas as pd
import datetime as dt
from collections import defaultdict

import setup
import qc_utils as utils

# TODO: revert
setup.SUBDAILY_OUT_DIR = setup.SUBDAILY_PROC_DIR

def get_files(diagnostics=False):
    """
    List all the files in the output directory which match the output format

    :param bool diagnostics: extra verbose output

    :returns: list    
    """
    file_extension = f'.qff{setup.OUT_COMPRESSION}'

    qff_files = [f for f in os.listdir(setup.SUBDAILY_OUT_DIR) if f.endswith(file_extension)]

    if diagnostics:
        print(f"Found {len(qff_files)} files in {setup.SUBDAILY_OUT_DIR}")
    return qff_files  # get_files


def process_files(qff_files, diagnostics=False):
    """
    Process each file in supplied list and build into a single dictionary
    
    :params list qff_files: input list of files to process
    :param bool diagnostics: extra verbose output
    
    :returns: dict    
    """
    # Initialize a dictionary to accumulate data frames for each year
    yearly_data = defaultdict(list)

    # Process each file
    for qfc, qfile in enumerate(qff_files):
        file_path = os.path.join(setup.SUBDAILY_OUT_DIR, qfile)
        
        # Read the .qff.gz file treating all columns as strings initially
        if setup.OUT_COMPRESSION == ".gz":
            df = pd.read_csv(file_path, sep='|', compression='gzip', dtype=str)
        else:
            df = pd.read_csv(file_path, sep='|', dtype=str)

        # Ensure the 'Year' column exists
        if 'Year' not in df.columns:
            print(f"'Year' column not found in {qfile}")
            #  TODO: need to write an error
            continue
        
        # Convert 'Year' column back to numeric
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        # Replace 'none' with empty string ('') in all columns
        df = df.applymap(lambda x: '' if x == 'none' else x)
        
        # Accumulate data frames by year
        for year, year_df in df.groupby('Year'):
            if not pd.isna(year):  # Ignore NaN years
                yearly_data[int(year)].append(year_df)
        
        if diagnostics:
            print(f"Processed {qfile} ({qfc+1}/{len(qff_files)})")

    return yearly_data  # process_files


def write_pqt(yearly_data, diagnostics=False):
    """
    Write each year to separate .parquet.gz files
    
    :param dict yearly_data: data in yearly form
    :param bool diagnostics: extra verbose output
  
    """
    # Define input and output directories
    output_dir = os.path.join(setup.ROOT_DIR, "pqt", setup.DATESTAMP)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # spin through each year
    for year, data_frames in yearly_data.items():

        if year >= utils.FIRST_YEAR and year <= dt.datetime.now().year:
            # Concatenate all data frames for the year
            combined_df = pd.concat(data_frames)
            
            # Save to .parquet.gz
            output_file = f"qff_{year}.parquet{setup.OUT_COMPRESSION}"
            output_path = os.path.join(output_dir, output_file)
            
            # Save to Parquet format
            if setup.OUT_COMPRESSION == ".gz":
                combined_df.to_parquet(output_path, compression='gzip', index=False, engine='pyarrow')
            else:
                combined_df.to_parquet(output_path, index=False, engine='pyarrow')
            
            if diagnostics:
                print(f"Written data for year {year} to {output_file}")


def main(diagnostics=False, clobber=False):
    """
    Main script.
    """

    qff_files = get_files(diagnostics=diagnostics)

    yearly_data = process_files(qff_files, diagnostics=diagnostics)

    write_pqt(yearly_data, diagnostics=diagnostics)

    print("All files processed successfully.")
    #  main


# ************************************************************************
if __name__ == "__main__":

    import argparse

    # set up keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics', dest='diagnostics', action='store_true', default=False,
                        help='Run diagnostics (will not write out file)')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False,
                        help='Overwrite output files if they exists.')

    args = parser.parse_args()

    main(diagnostics=args.diagnostics,
         clobber=args.clobber,
         )

#************************************************************************
