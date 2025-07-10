Creating Python Environment on ICHEC AWS instance
=================================================

The QC requires a specific Python environment to run.  This is done
with the ``conda`` package, both to build a specific Python environment,
and also to manage updates and versions of specific libraries.

Setting up conda
----------------

On AWS, you need to first install ``miniconda`` to allow access and creation
of the ``conda`` executables. This should update your
``.bashrc`` file.  Run the following 4 commands, answering "yes" to any questions
that arise.:

.. code:: console

   mkdir -p ~/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -u -p ~/miniconda3
   rm ~/miniconda3/miniconda.sh

You will then need to open a new terminal (or log out and back in again) to
give access to the tools.

Building the Environment
------------------------

If this is the first build, then you need to build the environment from the supplied ``environment.lock`` file.:

.. code:: console

    conda create --name glamod_QC --file environment.lock


This can take a while.  Once that has completed, you can load the environment with:

.. code:: console

    conda activate glamod_QC

And to deactivate:

.. code:: console

    conda deactivate

All the necessary Python libraries should have been installed.

.. note::
    The ``yaml`` files available in the repository are used to update the environment.
    This will be explained in a :doc:`how-to guide <../howto/index>`.
    For now, always use the lock environment as this is exactly reproducible.

.. note::
    There is also an oler ``make_venv.bash`` script, using the
    ``qc_venv_requirements.txt`` file, but this hasn't been tested in a while.