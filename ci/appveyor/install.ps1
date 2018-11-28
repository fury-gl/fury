# Powershell Install script

  "set PATH=%PYTHON%;%PYTHON%\\Scripts;%PATH%"
  ps: $env:PIPI = "pip install $env:EXTRA_PIP_FLAGS"
  echo %PIPI%
  # Check that we have the expected version and architecture for Python
  "python --version"
  ps: $env:PYTHON_ARCH = python -c "import struct; print(struct.calcsize('P') * 8)"
  ps: $env:PYTHON_VERSION = python -c "import platform;print(platform.python_version())"
  cmd: echo %PYTHON_VERSION% %PYTHON_ARCH%

  ps: |
        if($env:PYTHON -match "conda")
        {
          conda update -yq conda
          Invoke-Expression "conda install -yq  pip $env:DEPENDS"
          pip install nibabel cvxpy scikit-learn
        }
        else
        {
          python -m pip install -U pip
          pip --version
          if($env:INSTALL_TYPE -match "requirements")
          {
            Invoke-Expression "$env:PIPI -r requirements.txt"
          }
          else
          {
            Invoke-Expression "$env:PIPI $env:DEPENDS"
          }
          Invoke-Expression "$env:PIPI nibabel matplotlib scikit-learn cvxpy"
        }
  "%CMD_IN_ENV% python setup.py build_ext --inplace"
  "%CMD_IN_ENV% %PIPI% --user -e ."