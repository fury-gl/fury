#!/bin/bash
set -ev

# Anaconda
export PATH=${ENV_DIR}/miniconda/bin:$PATH
hash -r
source activate testenv

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

if [[ "${BUILD_DOCS}" == "1" ]]; then
  make -C docs html  # Build the documentation.
fi

#   - flake8 --max-line-length=115  # Enforce code style (but relax line length limit a bit).