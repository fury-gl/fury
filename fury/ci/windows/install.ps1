# Powershell Install script

if($env:INSTALL_TYPE -match "conda")
{
  # Get Anaconda path
  Write-Output "Conda path: $env:CONDA\Scripts"
  #gci env:*

  Invoke-Expression "conda config --set always_yes yes --set changeps1 no"
  Invoke-Expression "conda update -yq conda"
  Invoke-Expression "conda install conda-build anaconda-client"
  Invoke-Expression "conda config --add channels conda-forge"
  Invoke-Expression "conda create -n testenv --yes python=$env:PYTHON_VERSION pip"
  Invoke-Expression "conda install -yq --name testenv --file requirements/default.txt"
  Invoke-Expression "conda install -yq --name testenv --file requirements/test.txt"
  if($env:OPTIONAL_DEPS)
  {
    Invoke-Expression "conda install -yq --name testenv  --file requirements/optional.txt"
  }
}
else
{
  $env:PIPI = "pip install $env:EXTRA_PIP_FLAGS"
  # Print and check this environment variable
  Write-Output "Pip command: $env:PIPI"
  Invoke-Expression "python -m pip install -U pip"
  Invoke-Expression "pip --version"
  Invoke-Expression "$env:PIPI -r requirements/default.txt"
  Invoke-Expression "$env:PIPI -r requirements/test.txt"
  if($env:OPTIONAL_DEPS)
  {
    Invoke-Expression "$env:PIPI -r requirements/optional.txt"
  }
}
