# Powershell Install script

Write-Host "Pip command: " + $env:PIPI
$env:PIPI = "pip install $env:EXTRA_PIP_FLAGS"
# Print and check this environment variable
Write-Host "Pip command: " + $env:PIPI


# Print and Install FURY
Write-Host "Install FURY"
Invoke-Expression "$env:PIPI --user -e ."

# Run tests
Write-Host "Run FURY tests"
Invoke-Expression "pytest -svv fury"