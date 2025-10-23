'''
Obtain summary of obs per year for all stations

summary_counts_per_year.py invoked by typing::

  python summary_counts_per_year.py --restart_id --end_id [--diagnostics]

Input arguments:

--restart_id        First station to process

--end_id            Last station to process

--diagnostics       [False] Verbose output

'''
import datetime as dt
import numpy as np

# internal utils
import utils
import io_utils as io
import setup
#************************************************************************

#************************************************************************
def get_summary(stage: str="N", restart_id: str="",
                end_id: str="", diagnostics: bool=False) -> None:
    """
    Main script.  Reads in station data, populates internal objects extracts counts per year.

    :param str stage: after which stage to run Internal, Neighbour
    :param str restart_id: which station to start on
    :param str end_id: which station to end on
    :param bool diagnostics: print extra material to screen
    """

    # process the station list
    station_list = utils.get_station_list(restart_id=restart_id, end_id=end_id)

    station_IDs = station_list.id

    yearly_counts = {}

    # now spin through each ID in the curtailed list
    for st, station_id in enumerate(station_IDs):
        print(f"{dt.datetime.now()} {station_id:11s} ({st+1}/{station_IDs.shape[0]})")


        #*************************
        # set up the stations
        station = utils.Station(station_id, station_list.latitude[st],
                                station_list.longitude[st], station_list.elevation[st])
        if diagnostics:
            print(station)

        try:
            if stage == "I":
                station, _ = io.read_station(setup.SUBDAILY_PROC_DIR /
                                             f"{station_id:11s}{setup.OUT_SUFFIX}{setup.OUT_COMPRESSION}", station)
            elif stage == "N":
                station, _ = io.read_station(setup.SUBDAILY_OUT_DIR /
                                             f"{station_id:11s}{setup.OUT_SUFFIX}{setup.OUT_COMPRESSION}", station)

        except OSError: # as e:
            # file missing, move on to next in sequence
            # io.write_error(station, "File Missing", error=str(e))
            continue
        except ValueError: # as e:
            # some issue in the raw file
            # io.write_error(station, "Error in input file", error=str(e))
            continue

        # some may have no data (for whatever reason)
        if station.times.shape[0] == 0:
            if diagnostics:
                print(f"No data in station {station.id}")
            # scoot onto next station
            # io.write_error(station, "No data in input file")
            continue

        unique_years = np.unique(station.years)
        year_counts = np.zeros(unique_years.shape).astype(int)

        # spin through each variable (might be heaviest lift)
        for var in setup.obs_var_list:
            obs_var = getattr(station, var)

            # spin through each year
            for y, year in enumerate(unique_years):
                locs, = np.where(station.years == year)

                # where obs and years intersect
                year_obs = obs_var.data[locs]
                # and just keep unflagged set
                year_counts[y] += [len(year_obs.compressed())]

        # store in dictionary
        for year, count in zip(unique_years, year_counts):
            yearly_counts[year] = yearly_counts.get(year, 0) + count


    # now print
    with open("summary_counts.txt", "w") as outfile:
        for key, value in sorted(yearly_counts.items(), key=lambda x: x[0]):
            outfile.write(f"{key} : {value}\n")

    return # get_summary

#************************************************************************
if __name__ == "__main__":

    import argparse

    # set up keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', dest='stage', action='store', default="N",
                        help='After what to run - Internal or Neighbour, default="N"')
    parser.add_argument('--restart_id', dest='restart_id', action='store', default="",
                        help='Restart ID for truncated run, default=""')
    parser.add_argument('--end_id', dest='end_id', action='store', default="",
                        help='End ID for truncated run, default=""')
    parser.add_argument('--diagnostics', dest='diagnostics', action='store_true', default=False,
                        help='Run diagnostics (will not write out file)')
    args = parser.parse_args()

    get_summary(stage=args.stage,
               restart_id=args.restart_id,
               end_id=args.end_id,
               diagnostics=args.diagnostics)

#************************************************************************
