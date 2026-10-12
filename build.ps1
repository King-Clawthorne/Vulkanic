$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-StrictMode -Version Latest

Push-Location -LiteralPath $PSScriptRoot
try {
    $cmakeExecutable = Get-Command cmake -CommandType Application -All |
        Where-Object {
            $versionOutput = & $_.Source --version
            $versionMatch = [regex]::Match($versionOutput, '^cmake version (?<version>\d+\.\d+(?:\.\d+)?)')
            $versionMatch.Success -and [version]$versionMatch.Groups['version'].Value -ge [version]'4.4'
        } | Select-Object -First 1
    if (-not $cmakeExecutable) { throw 'CMake 4.4 or later is required. Add it to PATH before running this script.' }

    $cppFiles = @(Get-ChildItem -LiteralPath 'src' -Filter '*.cpp' -File | ForEach-Object { $_.FullName })
    if ($cppFiles.Count -eq 0) { throw 'No C++ source files were found under src.' }
    clang-format --style=file -i $cppFiles

    if (-not (Test-Path -LiteralPath 'build/build.ninja')) { & $cmakeExecutable.Source --preset windows }
    & $cmakeExecutable.Source --build --preset windows --parallel
    clang-tidy -p build $cppFiles
} finally { Pop-Location }

Write-Output "`nBuild completed successfully."
