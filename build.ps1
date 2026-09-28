$ErrorActionPreference = "Stop"
# Treat non-zero exit codes from native tools such as CMake as PowerShell errors.
$PSNativeCommandUseErrorActionPreference = $true
Set-StrictMode -Version Latest

# Resolve paths relative to this script so it can be launched from any directory.
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    # This helper intentionally performs a clean configure and build.
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    cmake --preset default
    cmake --build --preset default --parallel
} finally {
    # Restore the caller's working directory even when configuration or build fails.
    Pop-Location
}

Write-Output "`nBuild completed successfully."
