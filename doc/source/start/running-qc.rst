Processing Files
================

There is a bash script which currently is the quickest and easiest to run the system
but the longer term intention is that a Cylc suite will perform this in due course.

Over the course of the service since 2017, a number of different compute resources have
been used.  Each of these had a different selection of tools available to enable job
submission and management.  Hence there are a number of scripts available which have been
used for this purpose.  In the absence of a proper workflow management tool (e.g. Cylc), these
spin through the stations, running the internal or external QC checks.  There are three character
switches for these scripts, and all are called in the same way.

.. tab-set::

    .. tab-item:: Taskfarm (ICHEC-Kay)

        The script runs both stages of the QC process by submitting
        batches of stations through to the Kay cluster on ICHEC using the ``taskfarm``
        facility.

        Script Name: ``run_qc_taskfarm.bash``

        * ``I`` / ``N`` to run the Internal or Neighbour checks
        * ``T`` / ``F`` to wait (True) or not (False) for upstream files to be present
        * ``C`` / ``S`` to overwrite (Clobber) or keep (Skip) existing output files.

        So an example run for the internal checks:

        .. code:: console

            bash run_qc_taskfarm.bash I F C

    .. tab-item:: Parallel (AWS)

        The script runs both stages of the QC process by
        submitting batches of stations through to the CPU cluster
        using the ``parallel`` facility.

        Script Name: ``run_qc_parallel.bash``

        * ``I`` / ``N`` to run the Internal or Neighbour checks
        * ``T`` / ``F`` to wait (True) or not (False) for upstream files to be present
        * ``C`` / ``S`` to overwrite (Clobber) or keep (Skip) existing output files.

        So an example run for the internal checks:

        .. code:: console

            bash run_qc_parallel.bash I F C


    .. tab-item:: SLURM (Spice)

        The script runs both stages of the QC process by
        submitting batches of stations through to the CPU cluster
        using the ``slurm`` job scheduler.

        Script Name: ``run_qc_checks.bash``

        * ``I`` / ``N`` to run the Internal or Neighbour checks
        * ``T`` / ``F`` to wait (True) or not (False) for upstream files to be present
        * ``C`` / ``S`` to overwrite (Clobber) or keep (Skip) existing output files.

        So an example run for the internal checks:

        .. code:: console

            bash run_qc_checks.bash I F C


General notes
-------------

The waiting option presumes that the ``mff`` files will be produced in the sequence they
are listed in the station list (see :doc:`Setup Configuration<../start/setup-configuration>`).  If a file is not present,
the script will sleep until it appears.  This allows the QC process to start before the
mingle+merge processes have completed.

Finally, you can choose whether to overwrite
existing output files, or to skip the processing step if they already exists.  There
is helptext for these switches as part of the script.

Once completed, this script also runs a checking process to provide
some summary information of the processing run, with station counts
and locations.  This can be called separately as
``check_if_processed.bash`` using the ``I`` / ``N`` switches.

The python scripts called by bash scripts have their
own options which can be set.  For the moment, the one which allows stored
values and thresholds from a previous run to be used (rather than calculated afresh) is
not active.  This option was written with the near-real-time updates in mind, however
has never been tested on e.g. a "diff" file.  To turn this on, you would need to edit
the section of the bash script which generates the job.


Additional Outputs
------------------

There is a set of maps which can be produced, to show the flagging rates
and counts for each station for each test.  The Kay job for this is
submitted via the ``plot_scripts_slurm.bash`` /
``plot_scripts_parallel.bash`` script using the ``sbatch`` or
``parallel`` command.

There is also a script
``metadata_scripts_slurm.bash`` / ``metadata_scripts_parallel.bash``
which produces some of the metadata files to support the output data.

