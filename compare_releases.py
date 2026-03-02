from pathlib import Path
import pandas as pd
import datetime as dt

# internal utils
import io_utils as io

infile = Path("/data/users/robert.dunn/Copernicus/Rel8.1/release_8.1_simple_summary.txt")

version_summary = pd.read_csv(infile, delimiter="\t", header=0, index_col=0)
version_summary.columns = version_summary.columns.str.lstrip(" ")

infile = Path("/data/users/robert.dunn/Copernicus/Rel8.1/release_8_simple_summary.txt")

previous_summary = pd.read_csv(infile, delimiter="\t", header=0, index_col=0)
previous_summary.columns = previous_summary.columns.str.lstrip(" ")

#*********************************
# Additions/Extensions/Changes - use version as source

with open("Additions.txt", "w") as outfile:
    n_additions = 0
    for record in version_summary.itertuples():

        try:
            previous = previous_summary.loc[record.Index]
        except KeyError:
            n_additions += 1
            outfile.write(f"{record.Index} added\n")

print(f"{n_additions} stations added to list")

with open("Changes.txt", "w") as outfile:
    n_extensions = 0
    for record in version_summary.itertuples():
        text_string = ""

        try:
            previous = previous_summary.loc[record.Index]
        except KeyError:
            # Station not in previous, move on
            continue
        current = version_summary.loc[record.Index]

        if not previous.equals(current):
            n_extensions += 1
            # There's a difference

            if previous.N_Record != current.N_Record:
                text_string += f"{(current.N_Record-previous.N_Record):+d} records, "

            try:
                previous_start = dt.datetime(previous.start_Y, previous.start_M, previous.start_D)
                previous_end = dt.datetime(previous.end_Y, previous.end_M, previous.end_D)
                current_start = dt.datetime(current.start_Y, current.start_M, current.start_D)
                current_end = dt.datetime(current.end_Y, current.end_M, current.end_D)
            except ValueError:
                continue


            if previous_start != current_start:
                text_string += f"Starts {(previous_start - current_start).days:+d}days earlier, "

            if previous_end != current_end:
                text_string += f"Ends {(current_end - previous_end).days:+d}days later"

            if text_string != "":
                outfile.write(f"{record.Index} {text_string}\n")

print(f"{n_extensions} stations changed in release")


with open("NoChanges.txt", "w") as outfile:
    n_nochange = 0
    for record in version_summary.itertuples():

        try:
            previous = previous_summary.loc[record.Index]
        except KeyError:
            # Station not in previous, move on
            continue

        current = version_summary.loc[record.Index]

        if previous.equals(current):
            n_nochange += 1
            outfile.write(f"{record.Index} has no change\n")

print(f"{n_nochange} stations have no change in release")


#**********************************
# Removals - use previous as source

with open("Removals.txt", "w") as outfile:
    n_removals = 0
    for record in previous_summary.itertuples():

        try:
            current = version_summary.loc[record.Index]
        except KeyError:
            n_removals += 1
            outfile.write(f"{record.Index} removed\n")


print(f"{n_removals} stations removed from list")
