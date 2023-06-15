#!/bin/bash
set -ev

if [ "$INSTALL_TYPE" == "pip" ]; then
  hash -r
else
  # Anaconda
  export PATH=${ENV_DIR}/miniconda/bin:$PATH
  hash -r
  source activate testenv
fi

# Install and test FURY
cd ${TRAVIS_BUILD_DIR}
python3 setup.py install
# Change folder
mkdir for_testing
cd for_testing
# We need the setup.cfg for the pytest settings
cp ../setup.cfg .
python3 -c "import fury; print(fury.__version__)"
error_code=0

if [[ "${COVERAGE}" == "1" ]]; then
  cp ../.coveragerc .;
  cp ../.codecov.yml .;
  # coverage run -m pytest -svv --pyargs fury  # Run the tests and check for test coverage.
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
    # pytest -svv --pyargs fury
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

if [[ "${BUILD_DOCS}" == "1" && "$TRAVIS_OS_NAME" != "osx" ]]; then
  # Build the documentation.
  # xvfb-run --server-args="-screen 0, 1920x1080x24" \
  cd ${TRAVIS_BUILD_DIR}
  make -C docs html
fi

exit $error_code
set +ev
