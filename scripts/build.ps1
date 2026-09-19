param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if ($Clean -and (Test-Path build)) { Remove-Item -Recurse -Force build }

    cmake --preset default
    if ($LASTEXITCODE -ne 0) { throw "Configure failed." }

    cmake --build --preset $Configuration.ToLower() --parallel
    if ($LASTEXITCODE -ne 0) { throw "Build failed." }
} finally {
    Pop-Location
}

Write-Host "`nBuild completed successfully."
