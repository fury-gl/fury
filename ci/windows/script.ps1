# Powershell Install script

if($env:INSTALL_TYPE -match "conda")
{
    Invoke-Expression "conda activate testenv"
}

$env:PIPI = "pip install $env:EXTRA_PIP_FLAGS"
# Print and check this environment variable
Write-Output "Pip command: $env:PIPI"


# Print and Install FURY
Write-Output "======================== Install FURY ========================"
Invoke-Expression "$env:PIPI --user -e ."

# Run tests
Write-Output "======================== Run FURY tests ========================"
Invoke-Expression "pytest -svv fury"
