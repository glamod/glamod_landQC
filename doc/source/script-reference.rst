Individual QC checks
====================

The python scripts (``intra_checks.py`` for the internal checks,
and ``inter_checks.py`` for the neighbour/buddy checks)
run through each station in turn, applying the relevant checks.
The top-level QC-check functions are documented below:

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

.. automodule:: qc_tests.neighbour_outlier
   :members: noc

.. automodule:: qc_tests.clean_up
   :members: mcu

