#!/bin/bash

echo "Activate virtual environment"
source ci/activate_env.sh

set -ex

echo "Run the tests"

# Change into an innocuous directory and find tests from installation
mkdir for_testing
cd for_testing

# No figure windows for mpl; quote to hide : from travis-ci yaml parsing
echo "backend : agg" > matplotlibrc
if [ "$COVERAGE" == "1" ] || [ "$COVERAGE" == true ]; then
    cp ../.coveragerc .;
    cp ../.codecov.yml .;
    # Run the tests and check for test coverage.
    coverage run -m pytest -svv --doctest-modules --verbose --durations=10 --pyargs fury   # Need to --doctest-modules flag
    coverage report -m  # Generate test coverage report.
    coverage xml  # Generate coverage report in xml format for codecov.
    # codecov  # Upload the report to codecov.
else
    # Threads issue so we run test on individual file
    pytest -svv --doctest-modules --verbose --durations=10 --pyargs fury # Need to --doctest-modules flag
fi

cd ..
ls .

set +ex
