$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-StrictMode -Version Latest

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw 'WSL 2 is required. Install WSL 2 and a Linux distribution before building.'
}

# Pass the project path as an argument so spaces and shell metacharacters stay literal.
$wslBuildScript = @'
set -eu
case "$(cat /proc/sys/kernel/osrelease)" in
    *microsoft-standard-WSL2*) ;;
    *) echo 'This build requires WSL 2.' >&2; exit 1 ;;
esac

command -v wslpath >/dev/null 2>&1 || { echo 'Missing WSL tool: wslpath' >&2; exit 1; }
for tool in cmake ninja gcc g++ glslc git pkg-config; do
    if ! command -v "$tool" >/dev/null 2>&1 || ! "$tool" --version >/dev/null 2>&1; then
        echo "Missing Linux build tool: $tool" >&2
        echo 'Install CMake 3.30+, Ninja, GCC 15+, Vulkan development files, glslc, Git, and pkg-config in WSL 2.' >&2
        exit 1
    fi
done

cd "$(wslpath -u "$1")"
cmake --preset wsl
cmake --build --preset wsl --parallel
'@

wsl.exe --exec sh -c $wslBuildScript sh $PSScriptRoot
Write-Output "`nWSL 2 build completed successfully."
