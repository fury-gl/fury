#!/bin/bash

echo "Activate virtual environment"
source ci/activate_env.sh

set -ex

echo "Run the tests"

# Change into an innocuous directory and find tests from installation
mkdir for_testing
cd for_testing
# We need the setup.cfg for the pytest settings
cp ../setup.cfg .
# No figure windows for mpl; quote to hide : from travis-ci yaml parsing
echo "backend : agg" > matplotlibrc
if [ "$COVERAGE" == "1" ] || [ "$COVERAGE" == true ]; then
    cp ../.coveragerc .;
    cp ../.codecov.yml .;
    # Run the tests and check for test coverage.
    # coverage run -m pytest -svv --verbose --durations=10 --pyargs fury   # Need to --doctest-modules flag
    for file in `find ../fury -name 'test_*.py' -print`;
    do
        coverage run -m -p pytest -svv $file;
        retVal=$?
        if [ $retVal -ne 0 ]; then
            echo "THE CURRENT ERROR CODE IS $retVal";
            error_code=1
        fi
    done
    coverage combine .
    coverage report -m  # Generate test coverage report.
    codecov  # Upload the report to codecov.
else
    # Threads issue so we run test on individual file
    # pytest -svv --verbose --durations=10 --pyargs fury # Need to --doctest-modules flag
    for file in `find ../fury -name 'test_*.py' -print`;
    do
      pytest -svv $file;
      retVal=$?
      if [ $retVal -ne 0 ]; then
      echo "THE CURRENT ERROR CODE IS $retVal";
      error_code=1
      fi
    done
fi

set +ex