# Powershell Install script

# Setup some environment variable 
$env:Path += ";$env:PYTHON; $env:PYTHON\\Scripts"
$env:PIPI = "pip install $env:EXTRA_PIP_FLAGS"

# Print and check this environment variable
Write-Host $env:PIPI

# Check that we have the expected version and architecture for Python
Invoke-Expression "python --version"

$env:PYTHON_ARCH = python -c "import struct; print(struct.calcsize('P') * 8)"
$env:PYTHON_VERSION = python -c "import platform;print(platform.python_version())"

Write-Host "Python version: " + $env:PYTHON_VERSION 
Write-Host "Python architecture: " + $env:PYTHON_ARCH

if($env:PYTHON -match "conda")
{
  conda update -yq conda
  Invoke-Expression "conda install -yq  pip"
  Invoke-Expression "conda install -yq --file requirements/default.txt"
  Invoke-Expression "conda install -yq --file requirements/test.txt"
  if($env:OPTIONAL_DEPS)
  {
    Invoke-Expression "conda install -yq --file requirements/optional.txt"
  }
}
else
{
  Invoke-Expression "python -m pip install -U pip"
  Invoke-Expression "pip --version"
  Invoke-Expression "$env:PIPI -r requirements/default.txt"
  Invoke-Expression "$env:PIPI -r requirements/test.txt"
  if($env:OPTIONAL_DEPS)
  {
    Invoke-Expression "$env:PIPI -r requirements/optional.txt"
  }
}
