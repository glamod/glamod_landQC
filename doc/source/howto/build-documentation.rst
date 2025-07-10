Building the documentation
==========================

To build this Sphinx documentation to include all the doc-strings from the scripts into a pretty html file.
If you have run this before, it's good practice to remove the previous build as indicated.:

.. code:: console

    conda activate glamod_QC
    cd doc
    rm -r build/
    make html


Then you can open the ``index.html`` in the `doc/build/html/` directory.