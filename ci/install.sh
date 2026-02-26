#!/bin/bash

echo "Activate virtual environment"
source ci/activate_env.sh

set -ex

PIPI="pip install --timeout=60"

if [ "$USE_PRE" == "1" ] || [ "$USE_PRE" == true ]; then
    PIPI="$PIPI --extra-index-url=$PRE_WHEELS --pre";
fi

#---------- FURY Installation -----------------

if [ "$INSTALL_TYPE" == "setup" ]; then
    $PIPI .
elif [ "$INSTALL_TYPE" == "pip" ]; then
    $PIPI .
elif [ "$INSTALL_TYPE" == "sdist" ]; then
    python -m pip install build
    python -m build --sdist
    $PIPI dist/*.tar.gz
elif [ "$INSTALL_TYPE" == "wheel" ]; then
    python -m pip install build
    python -m build --wheel
    $PIPI dist/*.whl
elif [ "$INSTALL_TYPE" == "requirements" ]; then
    $PIPI -r requirements.txt
    $PIPI .
elif [ "$INSTALL_TYPE" == "conda" ]; then
    $PIPI .
fi

set +ex
