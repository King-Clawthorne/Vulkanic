$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
Set-StrictMode -Version Latest

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    cmake --preset default
    cmake --build --preset default --parallel
} finally {
    Pop-Location
}

Write-Host "`nBuild completed successfully."
