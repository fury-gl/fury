# Powershell Install script

$env:PIPI = "pip install $env:EXTRA_PIP_FLAGS"
# Print and check this environment variable
Write-Output "Pip command: " + $env:PIPI


# Print and Install FURY
Write-Output "Install FURY"
Invoke-Expression "$env:PIPI --user -e ."

# Run tests
Write-Output "Run FURY tests"
pytest -svv fury
# try {
#     Invoke-Expression "pytest -svv fury"
# } catch {
#     Write-Output "Error, Exception "
#     Write-Host $_
#     exit 1
# }
