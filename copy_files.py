#!/usr/bin/env python
'''
io_utils - contains scripts for read/write of main files
'''
import os
import shutil
import glob

#*********************************************
def copy_tree(source, destination, diagnostics=False):
    """
    Perform local copy from networked storage to working area 
        (e.g. GWS to /work/scratch )
        
    Automatically wipes and clobbers

    :param str source: source directory
    :param str destination: destination directory
    :param bool diagnostics: verbose output
    """

    # remove entire directory
    if os.path.exists(destination):
        try:
            shutil.rmtree(destination)
        except OSError:
            # already removed
            pass
        # don't need to make as copytree will do
        if diagnostics:
            print("{} removed".format(destination))
        

    # copy entire tree
    shutil.copytree(source, destination)
    # ensure update of timestamps
    for root, diry, files in os.walk(destination):
        for fname in files:
            os.utime(os.path.join(root, fname), None)

    if diagnostics:
        print("copied {} to {}".format(source, destination))
    
    return # copy_tree


#*********************************************
def copy_files(source, destination, extension="", clobber=True, wipe=True, diagnostics=False):
    """
    Perform local copy from networked storage to working area 
        (e.g. GWS/file.txt to /work/scratch/file.txt )

    :param str source: source directory
    :param str destination: destination directory
    :param bool clobber: overwrite
    :param bool wipe: clean out destination in advance of copy
    :param str extension: optional filename extension
    :param bool diagnostics: verbose output
    """

    # remove entire directory and recreate as blank
    if wipe:
        shutil.rmtree(destination)
        if not os.path.exists(destination):
            os.mkdir(destination)
        
    # for each file at a time
    for filename in glob.glob(r'{}*{}'.format(os.path.expanduser(source), extension)):

        if not os.path.exists(os.path.join(destination, filename.split("/")[-1])):
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
                    print(" exists".format(filename))
    
        # force update of timestamps
        os.utime(os.path.join(destination, filename.split("/")[-1]), None)

    return # copy_files
