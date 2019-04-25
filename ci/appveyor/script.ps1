# Powershell Install script

$env:PIPI = "pip install $env:EXTRA_PIP_FLAGS"
# Print and check this environment variable
Write-Output "Pip command: $env:PIPI"


# Print and Install FURY
Write-Output "======================== Install FURY ========================"
$env:PIPI --user -e .

# Run tests
Write-Output "======================== Run FURY tests ========================"
pytest -svv fury