.. C3S 311a Lot 2 QC documentation master file, created by
   sphinx-quickstart on Wed Jul 31 14:46:51 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

C3S 311a Lot 2 sub-daily QC suite documentation
===============================================

Introduction
------------

This is the documentation for the QC suite developed for the
Copernicus Climate Change Service (C3S) providing access
to in-situ observations.  The service's homepage can be found `here <https://climate.copernicus.eu/global-land-and-marine-observations-database>`_.

This QC suite has been developed for use on multi-variate,
station-based, sub-daily data, building on the HadISD quality control
software (`Dunn et al, 2012 <http://www.clim-past.net/8/1649/2012/cp-8-1649-2012.html>`_, `Dunn et al, 2016 <http://www.geosci-instrum-method-data-syst.net/5/473/2016/>`_, `Dunn 2019 <https://www.metoffice.gov.uk/learning/library/publications/science/climate-science-technical-notes>`_).  However,
although using the HadISD tests and codes as inspiration, a number of
requirements for this service mean the code base has been written from
scratch.

Python Environment on Kay/Bastion
---------------------------------

The QC requires a specific Python environment to run.  In the past this has been done with ``venv`` but the functionality and support on Kay with ``conda`` is better.  

On Kay, you need to load the module to allow access to ``conda``::

  module load conda

On Bastion, you need to install miniconda (which should update your
``.bashrc`` file)::

  mkdir -p ~/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -u -p ~/miniconda3
  rm ~/miniconda3/miniconda.sh

Then you need to build the environment from the supplied ``yml`` file::

  conda env create --name glamod_QC --file=environment.yml

This can take a while.  Once that has completed, you can load the environment with ::

  conda activate glamod_QC

And to deactivate::

  conda deactivate

All the necessary Python libraries should have been installed.
There is also an oler ``make_venv.bash`` script, using the
``qc_venv_requirements.txt`` file, but this hasn't been tested in a while.

Building the documentation
--------------------------

To build this Sphinx documentation to include all the doc-strings from the scripts into a pretty html file::

  [module load conda]
  conda activate glamod_QC
  cd doc
  make html

Then you can open the ``index.html`` in the `doc/build/html/` directory.


Configuring the QC
------------------

Configuration files handle the over-arching settings for the code,
including setting the paths to the files and the statistics and
thresholds to use.  The configuration.txt file contains::

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
  out_compression = .gz
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

The input "mff" (merged file format) files are in ``mff`` and the relevant sub-directory ``mff_version``, so the combination of these two entries give the location of the mff files.

The QC process produces a whole set of output files, the root directory of which is defined by ``root``.  The ``version`` sets the name of the subdirectory for each processing step, to keep individual runs separate.  This has usually been a datestamp.  Other entries give the directories as follows:

* ``proc`` The intermediate processed files, after internal but before buddy checks
* ``qff`` The final qff (QC'd file format) files
* ``config`` The location of the config files [see below]
* ``flag`` The flag files contining summary flagging rates and counts for each test for each variable
* ``images`` Any stored plots for an individual run are placed here, usually summary maps.
* ``errors`` Outputs from any managed errors within the scripts are here (empty files, unreadable inputs etc).
* ``logs`` Logs from the cluster processes (Taskfarm).

One final directory is as a subdirectory of the ``qff`` directory, called ``bad_stations``, which contains those output files from stations with high flagging rates or with other features that means they are withheld from further processing.

A number of files are required by this QC suite.  A station list, the variables to process, and a set of logic check values.  The ``station_list`` is the full path to the station list used/produced by the mingle process, containing IDs and locational metadata in a fixed-width format.

The QC can be configured to run using various statistics, which can be selected in the configuration file - there are two for the central tendency (``mean`` or ``median``) and three for the spread (``stdev``, ``iqr``, and ``mad`` [Median Absolute Deviation]).  The default settings are as per this repository.

There are a couple of thresholds which are used for when deciding whether to create a distribution or not (``min_data_count``) or for when the proportion of flags is classed as high (``high_flag_proportion``).

Finally, for the buddy/neighbour checks, there are a number of settings for selecting the neighbours (``max_distance`` and ``max_vertical_separation``), how many are selected (``max_number``), the filename to store the information (``filename``) and the minimum number needed for the tests to run (``minimum_number``).

[Note, the ``venv`` directory is kept as an input incase any future implementation uses venv rather than conda.]

Processing Files
----------------

There is a quick bash script which currently is the quickest and easiest to run the system but the longer term intention is that Rose/Cylc will perform this in due course.

On Kay
^^^^^^
The ``run_qc_taskfarm.bash`` runs both stages of the QC process by submitting batches of stations through to the Kay cluster on ICHEC using the taskfarm facility.  There are three character switches for this script:

* ``I`` / ``N`` to run the Internal or Neighbour checks
* ``T`` / ``F`` to wait (True) or not (False) for upstream files to be present
* ``C`` / ``S`` to overwrite (Clobber) or keep (Skip) existing output files.

So an example run for the internal checks::

  bash run_qc_taskfarm.bash I F C


On Bastion
^^^^^^^^^^

The ``run_qc_parallel.bash`` runs both stages of the QC process by
submitting batches of stations through to the Bastion CPU cluster
using the ``parallel`` facility.  There are three character switches for this script:

* ``I`` / ``N`` to run the Internal or Neighbour checks
* ``T`` / ``F`` to wait (True) or not (False) for upstream files to be present
* ``C`` / ``S`` to overwrite (Clobber) or keep (Skip) existing output files.

So an example run for the internal checks::

  bash run_qc_parallel.bash I F C

General notes
^^^^^^^^^^^^^

The waiting option presumes that the mff files will be produced in the sequence they are listed in the station list (see the configuration file).  If a file is not present, the script will sleep until it appears.  This allows the QC process to start before the mingle+merge processes have completed.  Finally, you can choose whether to overwrite existing output files, or to skip the processing step if they already exists.  There is helptext for these switches as part of the script.

Once completed, this script also runs a checking process to provide
some summary information of the processing run, with station counts
and locations.  This can be called separately as
``check_if_processed.bash`` using the ``I`` / ``N`` switches. There is
also a set of maps which can be produced, to show the flagging rates
and counts for each station for each test.  The Kay job for this is
submitted via the ``plot_scripts_slurm.bash`` /
``plot_scripts_parallel.bash`` script using the ``sbatch`` or
``parallel`` command.  There is also a script
``metadata_scripts_slurm.bash`` / ``metadata_scripts_parallel.bash``
which produces some of the metadata files to support the output data. 

The python scripts called by ``run_qc_taskfarm.bash`` / ``run_qc_parallel.bash`` have their own options which can be set (see below).  For the moment, the one which allows stored values and thresholds from a previous run to be used (rather than calculated afresh) is not active.  This option was written with the near-real-time updates in mind, however has never been tested on e.g. a "diff" file.  To turn this on, you would need to edit the section of the bash script which generates the job.


Individual scripts
------------------

The python scripts (``intra_checks.py`` for the internal checks, and ``inter_checks.py`` for the neighbour/buddy checks) run through each station in turn, applying the relevant tests.  These, and the top-level test functions are documented below:

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

.. automodule:: qc_tests.precision
   :members: pcc

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
