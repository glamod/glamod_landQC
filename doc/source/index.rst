.. C3S 311a Lot 2 QC documentation master file, created by
   sphinx-quickstart on Wed Jul 31 14:46:51 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

C3S 311a Lot 2 QC suite documentation
=============================================

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

The QC requires a specific Python environment to run.  The standard `JasPy <https://help.jasmin.ac.uk/article/4489-python-virtual-environments>`_ environment, although giving access to Python 3, does not contain all the packages.  Hence you need to set up another one.  Full instructions for JASMIN virtual environments are given `here <https://help.jasmin.ac.uk/article/4489-python-virtual-environments>`_.

Firstly, you need to load the JasPy environment to ensure that we're working in Python3.::

  module load jaspy

Then create your own virtual environment, using the command which ensures all the system packages are available from the outset.::

  virtualenv --system-site-packages venv

At this point you can unload JasPy:
  
  module unload jaspy

But to be safe, if you log out and back in again, that should clear that session.

Now, to activate the virtual environment::

  source venv/bin/activate

And to deactivate::

  deactivate

Now that you'll have a virtual environment, you'll need to install all the versions of the software that the QC takes to run.  The pip package takes care of all of that.  So after activating::

  pip install -r venv_requirements.txt

Which could take a while.


Running the QC
--------------

Configuration files handle the over-arching settings for the code,
including setting the paths to the files and the statistics and
thresholds to use.::

  [PATHS]
  root = /gws/nopw/j04/c3s311a_lot2/data/level1/land/
  mff = level1b_sub_daily_data/
  mff_version = menne_prelim_mff/
  qff = level1c_sub_daily_data/
  qff_version = v20191126/
  config = level1c_sub_daily_data_configs/
  images = level1c_sub_daily_data_plots/
  errors = level1c_sub_daily_data_errors/
  [FILES]
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

There is a quick bash script which currently is the quickest and easiest to run the system (please ask for details) but the intention is that Rose/Cylc will perform this in due coourse.

Processing Files
----------------

The main script runs through each station in turn, calling each test
in turn.  There will be buddy checks in future releases.

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
