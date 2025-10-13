Configuring the QC
==================

Configuration files handle the over-arching settings for the code,
including setting the paths to the files and the statistics and
thresholds to use.  The ``configuration.txt`` file contains:

.. code-block:: bash

    [PATHS]
    mff = /ichec/work/glamod/merge/
    mff_version = files/
    root = /ichec/work/glamod/data/level1/land/
    version = vYYYYMMDD
    proc = level1b1_sub_daily_data/
    qff = level1c_sub_daily_data/
    config = level1c_sub_daily_data_configs/
    flags = level1c_sub_daily_data_flags/
    images = level1c_sub_daily_data_plots/
    errors = level1c_sub_daily_data_errors/
    metadata = level1c_sub_daily_data_metadata/
    logs = level1c_sub_daily_data_logs/
    venvdir = .
    [FILES]
    station_list = /ichec/work/glamod/data/level1/land/level1b_sub_daily_data/stnlist/ghcnh-station-list-rel6.txt
    station_full_list = ghcnh_station_list.txt
    inventory = ghcnd_inventory.txt
    variables = obs_variables.json
    logic = logic_config.json
    in_compression =
    in_format = psv
    in_suffix = .psv
    out_compression = .gz
    out_format = psv
    out_suffix = .qff
    [STATISTICS]
    mean = False
    median = True
    stdev = False
    iqr = True
    mad = False
    [THRESHOLDS]
    min_data_count = 120
    high_flag_proportion = 0.2
    [NEIGHBOURS]
    max_distance = 500
    max_vertical_separation = 200
    max_number = 20
    filename = neighbours.txt
    minimum_number = 3
    [MISC]
    email = your.email@domain.com

The input ".mff" (merged file format) files are in ``mff`` and the relevant
sub-directory ``mff_version``, so the combination of these
two entries give the location of the mff files.

The ``in_compression`` and ``out_compression`` gives any compression applied to the input
or output files as an additional suffix (e.g. ``.gz`` or ``.zip`` [the ``.`` is necessary here]).
The ``in_format`` and ``out_format`` entries give the format of the input (mff) and output (qff)
files.  This can be in the form of ``psv`` [Pipe separated files (like csvs)] or ``pqt`` [Parquet files].
Finally, the ``in_suffix`` and ``out_suffix`` gives the filename extension in case this differs
from the format (e.g. use of ``mff`` or ``parquet``).


The QC process produces a whole set of output files, the root directory of which
is defined by ``root``.  The ``version`` sets the name of the subdirectory for each
processing step, to keep individual runs separate.  This has usually been a
datestamp.  Other entries give the directories as follows:

* ``proc`` The intermediate processed files, after internal but before buddy checks
* ``qff`` The final qff (QC'd file format) files
* ``config`` The location of the config files [see below]
* ``flag`` The flag files contining summary flagging rates and counts for each test for each variable
* ``images`` Any stored plots for an individual run are placed here, usually summary maps.
* ``errors`` Outputs from any managed errors within the scripts are here (empty files, unreadable inputs etc).
* ``logs`` Logs from the cluster processes (Taskfarm).

One final directory is as a subdirectory of the ``qff`` directory, called ``bad_stations``,
hich contains those output files from stations with high flagging rates or with other
features that means they are withheld from further processing.

A number of files are required by this QC suite.  A station list, the variables to process,
and a set of logic check values.  The ``station_list`` is the full path to the station list
used/produced by the mingle process, containing IDs and locational metadata in a
fixed-width format.  The ``station_full_list`` and ``inventory`` entries are for files
produced at the end of the processing chain, which emulate the ISD format for station
lists and station inventories.

The QC can be configured to run using various statistics, which can be selected in the
configuration file - there are two for the central tendency (``mean`` or ``median``)
and three for the spread (``stdev``, ``iqr``, and ``mad`` [Median Absolute Deviation]).
The default settings are as per this repository.

There are a couple of thresholds which are used for when deciding whether to create a
distribution or not (``min_data_count``) or for when the proportion of flags is
classed as high (``high_flag_proportion``).

Finally, for the buddy/neighbour checks, there are a number of settings for
selecting the neighbours (``max_distance`` and ``max_vertical_separation``),
how many are selected (``max_number``), the filename to store the information
(``filename``) and the minimum number needed for the tests to run (``minimum_number``).

[Note, the ``venv`` directory is kept as an input incase any future
implementation uses venv rather than conda.]