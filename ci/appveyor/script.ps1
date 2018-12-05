# Powershell Install script

  
# Print and Install FURY
Write-Host "Install FURY"
Invoke-Expression "$env:PIPI install ."

# Run tests
Write-Host "Run FURY tests"
Invoke-Expression "pytest -svv fury"