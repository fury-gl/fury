#!/bin/bash
set -ev

# Anaconda
export PATH=${ENV_DIR}/miniconda/bin:$PATH
hash -r
source activate testenv
conda install --yes --file ${TRAVIS_BUILD_DIR}/requirements/tests.txt

# Install and test FURY
cd ${TRAVIS_BUILD_DIR}
pip install .
pytest -svv fury

#   - coverage run -m pytest -svv fury  # Run the tests and check for test coverage.
#   - coverage report -m  # Generate test coverage report.
#   - codecov  # Upload the report to codecov.
#   - flake8 --max-line-length=115  # Enforce code style (but relax line length limit a bit).
#   - make -C docs html  # Build the documentation.