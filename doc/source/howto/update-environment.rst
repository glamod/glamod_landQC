Updating the Environment
========================

If you update package versions, e.g. to address security concerns or obtain extra
functionality, then you will need to remake the environment, test that it has no
inadvertent impacts, then remake lock file and save to the repository:

Do do this, we:

1. Remove development environment
2. Modify the ``environment.yml`` file to include new or updated packages.
3. Create a new environment using the updated ``environment.yml`` file.
4. Verify that the new environment functions as expected.
5. Generate an exact ``environment.lock`` file to ensure consistency for all developers.
6. Clean up and rebuild from new ``lock``.

.. note::

    Even if no changes are made to the ``environment.yml`` file, the resulting
    environment may still be updated as conda will look for the latest versions of
    packages if no exact version number is specified in the yml.

Step-By-Step
------------

1. Remove Environment
^^^^^^^^^^^^^^^^^^^^^

To update environment, make sure you are in the conda base:

.. code:: bash

    conda deactivate

Remove any previous environments used for the updating the QC environment.
The environment used for testing updating is called in the yml is called
``glamod_QC_dev``.


.. code:: bash

    conda env remove -n glamod_QC_dev

If no environment with the specified name exists, this commands will have no effect.



2. Update Requirements
^^^^^^^^^^^^^^^^^^^^^^

If needed, edit the ``environment.yml`` file.

Even if no changes are made, the new build will look for the latest versions of packages
listed in the ``.yml`` file.


3. Create New Environment
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: console

    conda env create --file environment.yml



4. Test New Environment
^^^^^^^^^^^^^^^^^^^^^^^

The above command creates an  environment called ``glamod_QC_dev``
(The name is specified in the file).


Activate and test the QC processes all work with the new environment.

.. code:: console

    conda activate glamod_QC_dev

    # run various tests....

E.g.

- Check that all tests pass
- Check that code runs on some example stations


.. caution::

    Be sure the new environment is fit for purpose before sharing it.


5. Create New Lock
^^^^^^^^^^^^^^^^^^

When happy the new environment is fit for purpose, create a new lock file
(overwrite the old one in the repository).  Using ``-e`` rather than ``--explict``
as don't want full urls given systems these environments are built on.


.. code:: console

    conda activate glamod_QC_dev
    conda list -e > environment.lock
    conda export --no-build > environment.yml
    conda export --no-build --from-history > environment-from-history.yml


You may need to check and update the environment name in the ``.lock`` file and
also remove the prefix if it contains sensitive information (e.g. user names and paths)

Create a PR for the new ``environment.yml``, ``environment-from-history.yml`` and ``environment.lock`` files!


.. tip::

    You could also update the old file used for the venv via

    .. code:: bash

        pip list --format=freeze > glamod_qc_venv_requirements.txt



6. Clean up
^^^^^^^^^^^

You can now delete your old locked environment and rebuild with the new lock.

.. code:: bash

    conda env remove -n glamod_QC



.. code:: bash

    conda create --name glamod_QC --file environment.lock



.. tip::

    Always try to work with the latest ``lock`` file in a repository. Ensure all
    developers are working with the exact ``environment.lock`` which should create an
    environment called ``glamod_QC``.