import pandas as pd
import calendar
import datetime as dt
import numpy as np

import setup

import setup

month_names = [c.upper() for c in calendar.month_abbr[:]]
month_names[0] = "ANN"

TODAY = dt.datetime.now()
TODAY_MONTH = dt.datetime.strftime(TODAY, "%B").upper()

def process_inventories() -> None:
    print("reading GHCNh inventories")

    # read in the inventories
    try:
        inventory = pd.read_csv(setup.SUBDAILY_METADATA_DIR / setup.INVENTORY,
                                delim_whitespace=True, skiprows=6, header=0)
    except OSError:
        pass

    try:
        previous_inventory = pd.read_csv(setup.SUBDAILY_METADATA_DIR.parent / setup.PREV_VERSION / setup.INVENTORY,
                                delim_whitespace=True, skiprows=6, header=0
            )
    except OSError:
        pass

    # now need to compare these

    # initially work on rows which are common to both
    common = inventory.merge(previous_inventory, on=["STATION", "YEAR"], how="inner")
    # https://stackoverflow.com/questions/60447865/subtracting-two-columns-named-in-certain-pattern
    difference = common.copy()
    difference.columns = difference.columns.str.rsplit('_', n=1, expand=True)
    difference = difference.xs('y', axis=1, level=1).sub(difference.xs('x', axis=1, level=1)).add_suffix('_diff')
    difference.insert(0, "YEAR", common["YEAR"])
    difference.insert(0, "STATION", common["STATION"])
    difference.columns = difference.columns.str.rstrip("_diff")

    # retain only rows where station/years have been added in latest release
    added = inventory.merge(previous_inventory, on=["STATION", "YEAR"],
                            how="left", indicator=True)
    added = added[added["_merge"] == "left_only"]
    added.drop(list(added.filter(regex = "_y")), axis=1, inplace=True)
    added.drop(columns=["_merge"], inplace=True)
#    added = added.replace(0, np.NaN).dropna()
    added.columns = added.columns.str.rstrip("_x")

    # now combine these two dataframes
    update_addition = pd.concat((difference, added))
    update_addition = update_addition.sort_values(["STATION", "YEAR"], ascending=[True, True])

    update_addition[["ANN","JAN", "FEB", "MAR", "APR", "MAY", "JUN",
             "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]] = update_addition[["ANN","JAN", "FEB", "MAR", "APR", "MAY", "JUN",
             "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]].astype(int)

    month_string = "\t".join([f"{c:<6s}" for c in month_names])
    outname = setup.SUBDAILY_METADATA_DIR / setup.CHANGED
    with open(outname, "w") as outfile:
        outfile.write("              *** GLOBAL HISTORICAL CLIMATE NETWORK HOURLY DATA INVENTORY ***\n")
        outfile.write("\n")
        outfile.write("THIS INVENTORY SHOWS THE CHANGE IN NUMBER OF WEATHER OBSERVATIONS BY STATION-YEAR-MONTH FOR BEGINNING OF RECORD\n")
        outfile.write(f"THROUGH {TODAY_MONTH} {TODAY.year}.  THE DATABASE CONTINUES TO BE UPDATED AND ENHANCED, AND THIS INVENTORY WILL BE \n")
        outfile.write("UPDATED ON A REGULAR BASIS. QUALITY CONTROL FLAGS HAVE NOT BEEN INCLUDED IN THESE COUNTS, ALL OBSERVATIONS.\n")
        outfile.write("\n")
        outfile.write(f"{'STATION':11s}\t{'YEAR':4s}\t{month_string:84s}\n")
        update_addition.to_csv(outfile, index=False, sep="\t", header=False)
        outfile.write("\n")

    # save only a condensed version of additions
    update_addition = update_addition.replace(0, np.NaN).dropna()
    update_addition[["ANN","JAN", "FEB", "MAR", "APR", "MAY", "JUN",
             "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]] = update_addition[["ANN","JAN", "FEB", "MAR", "APR", "MAY", "JUN",
             "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]].astype(int)
    outname = setup.SUBDAILY_METADATA_DIR / setup.CHANGED.stem + "_compressed" + setup.CHANGED.suffix
    with open(outname, "w") as outfile:
        outfile.write("              *** GLOBAL HISTORICAL CLIMATE NETWORK HOURLY DATA INVENTORY ***\n")
        outfile.write("\n")
        outfile.write("THIS INVENTORY SHOWS THE CHANGE IN NUMBER OF WEATHER OBSERVATIONS BY STATION-YEAR-MONTH FOR BEGINNING OF RECORD\n")
        outfile.write(f"THROUGH {TODAY_MONTH} {TODAY.year}.  THE DATABASE CONTINUES TO BE UPDATED AND ENHANCED, AND THIS INVENTORY WILL BE \n")
        outfile.write("UPDATED ON A REGULAR BASIS. QUALITY CONTROL FLAGS HAVE NOT BEEN INCLUDED IN THESE COUNTS, ALL OBSERVATIONS.\n")
        outfile.write("\n")
        outfile.write(f"{'STATION':11s}\t{'YEAR':4s}\t{month_string:84s}\n")
        update_addition.to_csv(outfile, index=False, sep="\t", header=False)
        outfile.write("\n")

    # retain only rows where station/years have been removed
    removed = inventory.merge(previous_inventory, on=["STATION", "YEAR"],
                              how="right", indicator=True)
    removed = removed[removed["_merge"] == "right_only"]
    removed.drop(list(removed.filter(regex = "_x")), axis=1, inplace=True)
    removed.drop(columns=["_merge"], inplace=True)
#    removed = removed.replace(0, np.NaN).dropna()
    removed.columns = removed.columns.str.rstrip("_y")
    # multiply by -1
    removed[["ANN","JAN", "FEB", "MAR", "APR", "MAY", "JUN",
             "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]] *= -1

    # save only a condensed version of removals
    removed = removed.replace(0, np.NaN).dropna()
    # want ints
    removed[["ANN","JAN", "FEB", "MAR", "APR", "MAY", "JUN",
             "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]] = removed[["ANN","JAN", "FEB", "MAR", "APR", "MAY", "JUN",
             "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]].astype(int)
    outname = setup.SUBDAILY_METADATA_DIR / setup.REMOVED
    with open(outname, "w") as outfile:
        outfile.write("              *** GLOBAL HISTORICAL CLIMATE NETWORK HOURLY DATA INVENTORY ***\n")
        outfile.write("\n")
        outfile.write("THIS INVENTORY SHOWS THE REDUCTION IN NUMBER OF WEATHER OBSERVATIONS BY STATION-YEAR-MONTH FOR BEGINNING OF RECORD\n")
        outfile.write(f"THROUGH {TODAY_MONTH} {TODAY.year} FROM PREVIOUS VERSION.  THE DATABASE CONTINUES TO BE UPDATED AND ENHANCED, AND THIS INVENTORY WILL BE \n")
        outfile.write("UPDATED ON A REGULAR BASIS. QUALITY CONTROL FLAGS HAVE NOT BEEN INCLUDED IN THESE COUNTS, ALL OBSERVATIONS.\n")
        outfile.write("\n")
        outfile.write(f"{'STATION':11s}\t{'YEAR':4s}\t{month_string:84s}\n")
        removed.to_csv(outfile, index=False, sep="\t", header=False)
        outfile.write("\n")


if __name__ == "__main__":

    process_inventories()
