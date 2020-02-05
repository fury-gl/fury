#!/bin/bash
set -ev

# Create deps dir
mkdir ${ENV_DIR}
cd ${ENV_DIR}

if [ "$INSTALL_TYPE" == "pip" ]; then
    PIPI="pip install $EXTRA_PIP_FLAGS"
    if [ -n "$USE_PRE" ]; then
        PIPI="$PIPI --find-links=$PRE_WHEELS --pre";
    fi
    $PIPI --upgrade setuptools pip
    $PIPI -r ${TRAVIS_BUILD_DIR}/requirements/default.txt
    $PIPI -r ${TRAVIS_BUILD_DIR}/requirements/test.txt
    if [[ "${OPTIONAL_DEPS}" == "1" ]]; then
        $PIPI -r ${TRAVIS_BUILD_DIR}/requirements/optional.txt
    fi
    if [[ "${BUILD_DOCS}" == "1" ]]; then
        $PIPI -r ${TRAVIS_BUILD_DIR}/requirements/docs.txt
    fi
else
    # Install Anaconda : Use the miniconda installer for faster download / install of conda itself
    if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
        # Does not exist by default so create this folder manually to avoid a Xvfb crash.
        mkdir /tmp/.X11-unix
        sudo chmod 1777 /tmp/.X11-unix
        sudo chown root /tmp/.X11-unix/
        # Start Xquartz -> more info at https://www.xquartz.org/
        Xquartz &
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;

    else
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
    fi
    chmod +x miniconda.sh && ./miniconda.sh -b -p ${ENV_DIR}/miniconda
    export PATH=${ENV_DIR}/miniconda/bin:$PATH
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update --yes -q conda
    conda install conda-build anaconda-client
    conda config --add channels conda-forge
    conda create -n testenv --yes python=$TRAVIS_PYTHON_VERSION pip
    source activate testenv
    conda install --yes --file ${TRAVIS_BUILD_DIR}/requirements/default.txt
    conda install --yes --file ${TRAVIS_BUILD_DIR}/requirements/test.txt
    if [[ "${OPTIONAL_DEPS}" == "1" ]]; then
        conda install --yes --file ${TRAVIS_BUILD_DIR}/requirements/optional.txt
    fi
    if [[ "${BUILD_DOCS}" == "1" ]]; then
        pip install -r  ${TRAVIS_BUILD_DIR}/requirements/docs.txt
    fi
fi
