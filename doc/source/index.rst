.. C3S 311a Lot 2 QC documentation master file, created by
   sphinx-quickstart on Wed Jul 31 14:46:51 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

C3S 311a Lot 2 sub-daily QC suite documentation
===============================================

Introduction
------------

This is the documentation for the QC suite developed for the
Copernicus Climate Change Service (C3S) "311a Lot 2", providing access
to in-situ observations.  The service's homepage can be found `here <https://climate.copernicus.eu/global-land-and-marine-observations-database>`_.

This QC suite has been developed for use on multi-variate,
station-based, sub-daily data, building on the HadISD quality control
software (`Dunn et al, 2012 <http://www.clim-past.net/8/1649/2012/cp-8-1649-2012.html>`_, `Dunn et al, 2016 <http://www.geosci-instrum-method-data-syst.net/5/473/2016/>`_, `Dunn 2019 <https://www.metoffice.gov.uk/learning/library/publications/science/climate-science-technical-notes>`_).  However,
although using the HadISD tests and codes as inspiration, a number of
requirements for this service mean the code base has been written from
scratch.

Python Environment on JASMIN
----------------------------

The QC requires a specific Python environment to run.  The standard `JasPy <https://help.jasmin.ac.uk/article/4489-python-virtual-environments>`_ environment, although giving access to Python 3, does not contain all the packages.  Hence you need to set up another one.  Full instructions for JASMIN virtual environments are `available <https://help.jasmin.ac.uk/article/4489-python-virtual-environments>`_.

A script is used to set up the bespoke virtual environment using the ``venv`` tool. 

Firstly, you need to edit the ``venvdir`` entry in the configuration file.  The default is called ``qc_venv``.::

  bash make_venv.bash

Thereafter you can activate the virtual environment::

  source qc_venv/bin/activate

And to deactivate::

  deactivate

All the necessary Python libraries should have been installed with the ``make_venv.bash`` script, but these are also listed in the ``qc_venv_requirements.txt`` file.

Configuring the QC
------------------

Configuration files handle the over-arching settings for the code,
including setting the paths to the files and the statistics and
thresholds to use.  The configuration.txt file contains::

  [PATHS]
  mff = /gws/nopw/j04/c3s311a_lot2/data/level1/land/level1b_sub_daily_data/
  mff_version = mff_latest/
  root = /gws/nopw/j04/c3s311a_lot2/data/level1/land/
  proc = level1b1_sub_daily_data/
  qff = level1c_sub_daily_data/
  config = level1c_sub_daily_data_configs/
  flags = level1c_sub_daily_data_flags/
  images = level1c_sub_daily_data_plots/
  errors = level1c_sub_daily_data_errors/
  venvdir = qc_venv
  [FILES]
  station_list = /gws/nopw/j04/c3s311a_lot2/data/level1/land/level1b_sub_daily_data/ghcnh-stations-20210116.txt
  variables = obs_variables.json
  logic = logic_config.json
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

The input "mff" (merged file format) files are in ``mff`` and the relevant sub-directory ``mff_version``, so the combination of these two entries give the location of the mff files.

The QC process produces a whole set of output files, the root directory of which is defined by ``root``.  The ``version`` sets the name of the subdirectory for each processing step, to keep individual runs separate.  This has usually been a datestamp.  Other entries give the directories as follows:

* ``proc`` The intermediate processed files, after internal but before buddy checks
* ``qff`` The final qff (QC'd file format) files
* ``config`` The location of the config files [see below]
* ``flag`` The flag files contining summary flagging rates and counts for each test for each variable
* ``images`` Any stored plots for an individual run are placed here, usually summary maps.
* ``errors`` Outputs from any managed errors within the scripts are here (empty files, unreadable inputs etc).

One final directory is as a subdirectory of the ``qff`` directory, called ``bad_stations``, which contains those output files from stations with high flagging rates or with other features that means they are withheld from further processing.

A number of files are required by this QC suite.  A station list, the variables to rprocess, and a set of logic check values.  The ``station_list`` is the full path to the station list used/produced by the mingle process, containing IDs and locational metadata in a fixed-width format.

The QC can be configured to run using various statistics, which can be selected in the configuration file - there are two for the central tendency (``mean`` or ``median``) and three for the spread (``stdev``, ``iqr``, and ``mad`` [Median Absolute Deviation]).  The default settings are as per this repository.

There are a couple of thresholds which are used for when deciding whether to create a distribution or not (``min_data_count``) or for when the proportion of flags is classed as high (``high_flag_proportion``).

Finally, for the buddy/neighbour checks, there are a number of settings for selectig the neighbours (``max_distance`` and ``max_vertical_separation``), how many are selected (``max_number``), the filename to store the information (``filename``) and the minimum number needed for the tests to run (``minimum_number``).


Processing Files
----------------

There is a quick bash script which currently is the quickest and easiest to run the system but the longer term intention is that Rose/Cylc will perform this in due course.

The ``run_qc_checks.bash`` runs both stages of the QC process by submitting one job per station to the JASMIN LOTUS cluster.  There are three character switches for this script:

* ``I`` / ``N`` to run the Internal or Neighbour checks
* ``T`` / ``F`` to wait (True) or not (False) for upstream files to be present
* ``C`` / ``S`` to overwrite (Clobber) or keep (Skip) existing output files.

The waiting option presumes that the mff files will be produced in the sequence they are listed in the station list (see the configuration file).  If a file is not present, the script will sleep until it appears.  This allows the QC process to start before the mingle+merge processes have completed.  Finally, you can choose whether to overwrite existing output files, or to skip the processing step if they already exists.  There is helptext for these switches as part of the script.

Once completed, this script also runs a checking process to provide some summary information of the processing run, with station counts and locations.  This can be called separately as ``check_if_processed.bash`` using the ``I`` / ``N`` switches. There is also a set of maps which can be produced, to show the flagging rates and counts for each station for each test.  The LOTUS job for this is submitted via the ``plots_lotus.bash`` script using the ``sbatch`` command.

The python scripts (``intra_checks.py`` for the internal checks, and ``inter_checks.py`` for the buddy checks) run through each station in turn, applying the relevant tests.  These, and the top-level test functions are documented below:

.. toctree::
   :maxdepth: 2

.. automodule:: intra_checks
   :members: run_checks

.. automodule:: qc_tests.logic_checks
   :members: lc

.. automodule:: qc_tests.odd_cluster
   :members: occ

.. automodule:: qc_tests.frequent
   :members: fvc

.. automodule:: qc_tests.diurnal
   :members: dcc

.. automodule:: qc_tests.distribution
   :members: dgc

.. automodule:: qc_tests.world_records
   :members: wrc

.. automodule:: qc_tests.streaks
   :members: rsc

.. automodule:: qc_tests.climatological
   :members: coc

.. automodule:: qc_tests.timestamp
   :members: tsc

.. automodule:: qc_tests.spike
   :members: sc

.. automodule:: qc_tests.humidity
   :members: hcc

.. automodule:: qc_tests.variance
   :members: evc

.. automodule:: qc_tests.pressure
   :members: pcc

.. automodule:: qc_tests.winds
   :members: wcc

.. automodule:: qc_tests.high_flag
   :members: hfr

.. automodule:: inter_checks
   :members: run_checks

.. automodule:: qc_tests.neighbour_outlier
   :members: noc

.. automodule:: qc_tests.clean_up
   :members: mcu



References
----------

HadISD version 3: monthly updates, RJH Dunn, Met Office Hadley Centre
Technical Note, #103, 2019, https://www.metoffice.gov.uk/learning/library/publications/science/climate-science-technical-notes

Expanding HadISD: quality-controlled, sub-daily station data from
1931, RJH Dunn, KM Willett, DE Parker, L Mitchell,
Geosci. Instrum. Method. Data Syst., 5, 473-491, (2016),
http://www.geosci-instrum-method-data-syst.net/5/473/2016/ 

HadISD: a quality controlled global synoptic report database for selected
variables at long-term stations from 1973-2010, RJH Dunn, KM Willett,
PW Thorne, EV Woolley, I Durre, A Dai, DE Parker, RS Vose, Climate of
the Past 8, 1649-1679 (2012),
http://www.clim-past.net/8/1649/2012/cp-8-1649-2012.html 
