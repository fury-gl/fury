#!/bin/bash

echo "Activate virtual environment"
source ci/activate_env.sh

set -ex

echo "Display Python version"
python -c "import sys; print(sys.version)"
python -m pip install -U pip setuptools>=30.3.0 wheel


echo "Install Dependencies"
if [ "$INSTALL_TYPE" == "conda" ]; then
	conda install -yq --name venv --file requirements/default.txt
    conda install -yq --name venv --file requirements/test.txt
    if [ "$DEPENDS" == "OPTIONAL_DEPS" ]; then conda install -yq --name venv --file requirements/optional.txt; fi
    if [ "$BUILD_DOCS" == "1" ]; then conda install -yq --name venv --file requirements/docs.txt; fi
else
    PIPI="pip install --timeout=60 "

    if [ "$USE_PRE" == "1" ] || [ "$USE_PRE" = true ]; then PIPI="$PIPI --extra-index-url=$PRE_WHEELS --pre"; fi

    $PIPI pytest
    $PIPI numpy

	$PIPI -r requirements/default.txt
    $PIPI -r requirements/test.txt

    if [ "$DEPENDS" == "OPTIONAL_DEPS" ]; then $PIPI -r requirements/optional.txt; fi
    if [ "$BUILD_DOCS" == "1" ]; then $PIPI -r requirements/docs.txt; fi
    if [ "$COVERAGE" == "1" ] || [ "$COVERAGE" = true ]; then pip install coverage coveralls codecov; fi
fi

set +ex
