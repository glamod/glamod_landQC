#!/bin/bash
set -x

cwd=`pwd`
# use configuration file to pull out paths &c
CONFIG_FILE="${cwd}/configuration.txt"
VENVDIR=$(grep "venvdir " "${CONFIG_FILE}" | awk -F'= ' '{print $2}')

echo "removing ${VWNVDIR}"
rm -rf ${VENVDIR}

echo "creating ${VENVDIR}"
module load jaspy
python3 -m venv ${VENVDIR}
source ${VENVDIR}/bin/activate
module unload jaspy

echo "populating ${VENVDIR}"
# do installs manually
pip install scitools-iris
# includes Cartopy, numpy, scipy, matplotlib
pip install pandas
pip install ipython
pip install reverse-geocoder
pip install alabaster
# as there are two cfunits packages, get the other one
pip install cfunits
pip install Sphinx
pip install sphinx-rtd-theme


deactivate
# END
