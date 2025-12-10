#!/usr/bin/env python
'''
copy_files - scripts to copy files and trees
'''
from pathlib import PurePath
import os
import shutil
import glob

#*********************************************
def copy_tree(source: PurePath, destination: PurePath, diagnostics: bool = False) -> None:
    """
    Perform local copy from networked storage to working area
        (e.g. GWS to /work/scratch )

    Automatically wipes and clobbers

    :param PurePath source: source directory
    :param PurePath destination: destination directory
    :param bool diagnostics: verbose output
    """

    # remove entire directory
    if destination.exists():
        try:
            shutil.rmtree(destination)
        except OSError:
            # already removed
            pass
        # don't need to make as copytree will do
        if diagnostics:
            print(f"{destination} removed")


    # copy entire tree
    shutil.copytree(source, destination)
    # ensure update of timestamps
    for root, _, files in destination.walk():
        for fname in files:
            os.utime(root / fname, None)

    if diagnostics:
        print(f"copied {source} to {destination}")

    # copy_tree


#*********************************************
def copy_files(source: PurePath, destination: PurePath, extension:str = "",
               clobber:bool = True, wipe: bool = True, diagnostics: bool = False) -> None:
    """
    Perform local copy from networked storage to working area
        (e.g. GWS/file.txt to /work/scratch/file.txt )

    :param PurePath source: source directory
    :param PurePath destination: destination directory
    :param bool clobber: overwrite
    :param bool wipe: clean out destination in advance of copy
    :param str extension: optional filename extension
    :param bool diagnostics: verbose output
    """

    # remove entire directory and recreate as blank
    if wipe:
        shutil.rmtree(destination)
        if not destination.exists():
            destination.mkdir()

    # for each file at a time
    for filename in glob.glob(fr'{source.expanduser()}*{extension}'):

        if not (destination / filename.split("/")[-1]).exists():
            # file doesn't exist, so copy
            shutil.copy(filename, destination)

            if diagnostics:
                print(filename)
        else:
            # file exists
            if clobber:
                # overwrite
                shutil.copy(filename, destination)

                if diagnostics:
                    print(filename)

            else:
                if diagnostics:
                    print(f"{filename} exists")

        # force update of timestamps
        os.utime(destination / filename.split("/")[-1], None)

    # copy_files
