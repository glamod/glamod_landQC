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
to in-situ observations.  The service's homepage can be found
`here <https://climate.copernicus.eu/global-land-and-marine-observations-database>`_.

This QC suite has been developed for use on multi-variate,
station-based, sub-daily data, building on the HadISD quality control
software (`Dunn et al, 2012 <http://www.clim-past.net/8/1649/2012/cp-8-1649-2012.html>`_,
`Dunn et al, 2016 <http://www.geosci-instrum-method-data-syst.net/5/473/2016/>`_,
`Dunn 2019 <https://www.metoffice.gov.uk/learning/library/publications/science/climate-science-technical-notes>`_).
However, although using the HadISD tests and codes as inspiration, a number of
requirements for this service mean the code base has been written from
scratch.








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
