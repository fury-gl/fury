#!/bin/bash
set -ev

# Anaconda
export PATH=${ENV_DIR}/miniconda/bin:$PATH
hash -r
conda activate testenv

# Install and test FURY
cd ${TRAVIS_BUILD_DIR}
pip install .
if [[ "${COVERAGE}" == "1" ]]; then
  coverage run -m pytest -svv fury  # Run the tests and check for test coverage.
  coverage report -m  # Generate test coverage report.
  codecov  # Upload the report to codecov.
else
    pytest -svv fury
fi

if [[ "${BUILD_DOCS}" == "1" && "$TRAVIS_OS_NAME" != "osx" ]]; then
  # Build the documentation.
  xvfb-run --server-args="-screen 0, 1920x1080x24" \
    make -C docs html  
fi
