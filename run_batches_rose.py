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
import random

# internal utils
import utils

# scripts to call
import intra_checks as intra
#import inter_checks as inter
#import copy_files


#*******************
def process_batches(batch: int, total: int) -> tuple[str, str]:
    '''
    Use batch number and the total number of batches to return
    the restart_id and end_id for the run.
    '''

    # get the most recent station list
    station_list = utils.get_station_list(restart_id="", end_id="")

    # and shuffle with a repeatable sequence for a given list of stations
    # so that long/large stations (USA etc) aren't all in the same batches
    order = np.arange(station_list.shape[0])
    PSEUDO_RANDOM_SEED=500
    random.Random(PSEUDO_RANDOM_SEED).shuffle(order)
    station_list=station_list[order]

    station_IDs = station_list.id

    # find indices in suitable spacing
    splits = np.linspace(0, len(station_list) - 1, total + 1).astype(int)

    # duplicate to get the start and end index
    starts = splits[:-1]
    ends = splits[1:] - 1 # as the scripts have inclusive not exclusive end points
    ends[-1] = ends[-1] + 1 # to ensure no out of range error

#    print(starts, ends, len(starts), len(ends))
    print(station_IDs.iloc[starts[batch]], station_IDs.iloc[ends[batch]])

    # return the restart_id and end_id for use
    return station_IDs.iloc[starts[batch]], station_IDs.iloc[ends[batch]] # process_batches

#************************************************************************
def run_intra_station_checks(restart_id: str, end_id: str, full: bool) -> None:
    '''
    Run intra-station checks with standard settings
    '''

    intra.run_checks(restart_id=restart_id, end_id=end_id, diagnostics=False, plots=False, full=full)

    # run_intra_station_checks


#************************************************************************
def run_inter_station_checks(restart_id: str, end_id: str, full: bool) -> None:
    '''
    Run inter-station (buddy/neighbour) checks with standard settings
    '''

#    inter.run_checks(restart_id=restart_id, end_id=end_id, diagnostics=False, plots=False, full=full)

    # run_inter_station_checks


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
