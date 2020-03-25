'''
Create batches of stations to run in parallel on e.g. LOTUS to allow Rose suite to schedule

run_batches_rose.py invoked by typing::

  python run_batches_rose.py --batch --total --stage [--full]

Input arguments:

--batch             Which batch

--total             Total number of batches

--stage             Which stage of the processing to run

--full              [False] Run a full reprocessing (recalculating thresholds) rather than reading from files

-'''

#************************************************************************
import argparse
import numpy as np

# internal utils
import setup
import qc_utils as utils

# scripts to call
import intra_checks as intra
import inter_checks as inter
import copy_files


#*******************
def process_batches(batch, total):
    '''
    Use batch number and the total number of batches to return
    the restart_id and end_id for the run.
    '''

    # get the most recent station list
    station_list = utils.get_station_list(restart_id="", end_id="")
    station_IDs = station_list.iloc[:, 0]

    # find indices in suitable spacing
    splits = np.linspace(0, len(station_list) - 1, total + 1).astype(int)

    # duplicate to get the start and end index
    starts = splits[:-1]
    ends = splits[1:] - 1 # as the scripts have inclusive not exclusive end points
    ends[-1] = ends[-1] + 1 # to ensure no out of range error

#    print(starts, ends, len(starts), len(ends))
    print(station_IDs[starts[batch]], station_IDs[ends[batch]])

    # return the restart_id and end_id for use
    return station_IDs[starts[batch]], station_IDs[ends[batch]] # process_batches

#************************************************************************
def run_intra_station_checks(restart_id, end_id, full):
    '''
    Run intra-station checks with standard settings
    '''

    if setup.runon == "scratch":
        # need to copy files
        copy_files.copy_tree(setup.SUBDAILY_ROOT_DIR, setup.SCRATCH_ROOT_DIR)

    intra.run_checks(restart_id=restart_id, end_id=end_id, diagnostics=False, plots=False, full=full)

    return # run_intra_station_checks

#************************************************************************
def run_inter_station_checks(restart_id, end_id, full):
    '''
    Run inter-station (buddy/neighbour) checks with standard settings
    '''

#    inter.run_checks(restart_id=restart_id, end_id=end_id, diagnostics=False, plots=False, full=full)

    return # run_inter_station_checks

#************************************************************************
if __name__ == "__main__":

    # set up keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', dest='batch', action='store', type=int, default=0,
                        help='batch number')
    parser.add_argument('--total', dest='total', action='store', type=int, default=0,
                        help='total number of batches')
    parser.add_argument('--stage', dest='stage', action='store', default="",
                        help='stage of QC to run')
    parser.add_argument('--full', dest='full', action='store_true', default=False,
                        help='Run full reprocessing rather than just an updat')

    args = parser.parse_args()

    restart_id, end_id = process_batches(args.batch, args.total)

    # intra-station checks
    if args.stage == "I":
        run_intra_station_checks(restart_id=restart_id,
                                 end_id=end_id,
                                 full=args.full)
    if args.stage == "N":
        run_inter_station_checks(restart_id=restart_id,
                                 end_id=end_id,
                                 full=args.full)

#************************************************************************
