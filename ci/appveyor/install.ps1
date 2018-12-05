# Powershell Install script


$env:PIPI = "pip install $env:EXTRA_PIP_FLAGS"
# Print and check this environment variable
Write-Host "Pip command: " + $env:PIPI

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
