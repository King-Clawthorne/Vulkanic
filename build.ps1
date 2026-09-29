$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-StrictMode -Version Latest

Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath 'build/build.ninja')) {
        cmake --preset windows
    }
    cmake --build --preset windows --parallel
} finally {
    Pop-Location
}

Write-Output "`nBuild completed successfully."
