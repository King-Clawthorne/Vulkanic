param(
    [string]$BuildDir = "cmake-build-ninja",
    [string]$Configuration = "Release",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ... [Resolve-VcVarsPath and Import-VcVars functions remain the same] ...

function Resolve-VcVarsPath {
    $candidates = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )
    foreach ($candidate in $candidates) { if (Test-Path $candidate) { return $candidate } }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installPath) {
            $resolved = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $resolved) { return $resolved }
        }
    }
    throw "Could not find vcvars64.bat."
}

function Import-VcVars {
    param([string]$vcvarsPath)
    Write-Host "Loading MSVC environment variables..." -NoNewline
    $envLines = & cmd.exe /c "call `"$vcvarsPath`" > NUL && set"
    foreach ($line in $envLines) {
        if ($line -match "^([^=]+)=(.*)$") { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process") }
    }
    Write-Host " Done."
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$vcvarsPath = Resolve-VcVarsPath
$buildPath = Join-Path $repoRoot $BuildDir

Import-VcVars -vcvarsPath $vcvarsPath

if ($Clean -and (Test-Path $buildPath)) {
    Remove-Item -Recurse -Force $buildPath
}

Write-Host "Configuring and building in: $BuildDir"

# 1. Start sccache server (it usually starts on its own, but this ensures it's ready)
$useSccache = $false
if (Get-Command sccache -ErrorAction SilentlyContinue) {
    Write-Host "sccache detected. Enabling compiler caching."
    sccache --start-server
    $useSccache = $true
} else {
    Write-Warning "sccache not found in PATH. Proceeding without compiler caching."
}

# 2. Configure project
# CMakeLists.txt already rewrites /Zi to /Z7 for MSVC builds so sccache can cache them.
$cmakeArgs = @(
    "-S", "$repoRoot",
    "-B", "$buildPath",
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$Configuration"
)
if ($useSccache) {
    $cmakeArgs += "-DCMAKE_C_COMPILER_LAUNCHER=sccache"
    $cmakeArgs += "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
}

& cmake @cmakeArgs

if ($LASTEXITCODE -ne 0) {
    throw "Configure failed."
}

# 3. Build
& cmake --build "$buildPath" --config "$Configuration" --parallel

if ($LASTEXITCODE -ne 0) {
    throw "Build failed."
}

# 4. Show cache stats so you can see the hits
if ($useSccache) {
    Write-Host "`n--- sccache Statistics ---"
    sccache --show-stats
}

Write-Host "`nBuild completed successfully."
