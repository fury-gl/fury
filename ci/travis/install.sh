#!/bin/bash
set -ev

# Create deps dir
mkdir ${DEPS_DIR}
cd ${DEPS_DIR}


#   # Install this package and the packages listed in requirements.txt.
#   - pip install .
#   # Install extra requirements for running tests and building docs.
#   - pip install -r requirements-dev.txt

# Install Anaconda : Use the miniconda installer for faster download / install of conda itself
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
else
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi

chmod +x miniconda.sh && ./miniconda.sh -b -p ${DEPS_DIR}/miniconda
export PATH=${DEPS_DIR}/miniconda/bin:$PATH
hash -r
conda config --set always_yes yes --set changeps1 no
conda update --yes -q conda
conda install conda-build anaconda-client
conda create -n testenv --yes python=$TRAVIS_PYTHON_VERSION pip mesa-utils
source activate testenv
conda install --yes --file ${TRAVIS_BUILD_DIR}/requirements/default.txt
conda install --yes --file ${TRAVIS_BUILD_DIR}/requirements/test.txt
if [[ "${OPTIONAL_DEPS}" == "1" ]]; then
    conda install --yes --file ${TRAVIS_BUILD_DIR}/requirements/optional.txt
fi
