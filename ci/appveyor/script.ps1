# Powershell Install script

  
# Print and Install FURY
Write-Host "Install FURY"
Invoke-Command { $env:PIPI --user -e .}

# Run tests
Write-Host "Run FURY tests"
Invoke-Command {pytest -svv fury}