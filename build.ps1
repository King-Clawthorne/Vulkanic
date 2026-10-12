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
    $buildPath = Join-Path $PSScriptRoot 'build'
    $buildNinjaPath = Join-Path $buildPath 'build.ninja'
    $compileCommandsPath = Join-Path $buildPath 'compile_commands.json'
    if (-not (Test-Path -LiteralPath $buildNinjaPath) -or -not (Test-Path -LiteralPath $compileCommandsPath)) { & $cmakeExecutable.Source --preset windows }
    if (-not (Test-Path -LiteralPath $compileCommandsPath)) { throw 'CMake did not generate build/compile_commands.json.' }
    & $cmakeExecutable.Source --build --preset windows --parallel
    $clangTidyExecutable = Get-Command clang-tidy -CommandType Application | Select-Object -First 1
    $maxParallelTidyProcesses = [Math]::Min($cppFiles.Count, [Environment]::ProcessorCount)
    $pendingTidyFiles = [System.Collections.Generic.Queue[string]]::new()
    foreach ($cppFile in $cppFiles) { $pendingTidyFiles.Enqueue($cppFile) }
    $runningTidyProcesses = [System.Collections.Generic.List[object]]::new()
    $tidyResults = [System.Collections.Generic.List[object]]::new()
    while ($pendingTidyFiles.Count -gt 0 -or $runningTidyProcesses.Count -gt 0) {
        while ($pendingTidyFiles.Count -gt 0 -and $runningTidyProcesses.Count -lt $maxParallelTidyProcesses) {
            $cppFile = $pendingTidyFiles.Dequeue()
            $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
            $startInfo.FileName = $clangTidyExecutable.Source
            $startInfo.WorkingDirectory = $PSScriptRoot
            $startInfo.UseShellExecute = $false
            $startInfo.RedirectStandardOutput = $true
            $startInfo.RedirectStandardError = $true
            [void]$startInfo.ArgumentList.Add('-p')
            [void]$startInfo.ArgumentList.Add($buildPath)
            [void]$startInfo.ArgumentList.Add($cppFile)
            Write-Host "Starting clang-tidy for $cppFile"
            $process = [System.Diagnostics.Process]::Start($startInfo)
            $runningTidyProcesses.Add([pscustomobject]@{
                File = $cppFile
                Process = $process
                Stdout = $process.StandardOutput.ReadToEndAsync()
                Stderr = $process.StandardError.ReadToEndAsync()
            })
        }

        $completedTidyProcess = $runningTidyProcesses | Where-Object { $_.Process.HasExited } | Select-Object -First 1
        if ($null -eq $completedTidyProcess) {
            Start-Sleep -Milliseconds 25
            continue
        }

        $completedTidyProcess.Process.WaitForExit()
        $stdout = $completedTidyProcess.Stdout.GetAwaiter().GetResult()
        $stderr = $completedTidyProcess.Stderr.GetAwaiter().GetResult()
        if ($stdout) { Write-Host $stdout -NoNewline }
        if ($stderr) { Write-Host $stderr -NoNewline }
        $tidyResults.Add([pscustomobject]@{ File = $completedTidyProcess.File; ExitCode = $completedTidyProcess.Process.ExitCode })
        $completedTidyProcess.Process.Dispose()
        [void]$runningTidyProcesses.Remove($completedTidyProcess)
    }

    $failedTidyResults = @($tidyResults | Where-Object { $_.ExitCode -ne 0 })
    if ($failedTidyResults.Count -gt 0) {
        $failures = $failedTidyResults | ForEach-Object { "$($_.File) (exit code $($_.ExitCode))" }
        throw "clang-tidy failed for $($failures -join ', ')."
    }
} finally { Pop-Location }

Write-Output "`nBuild completed successfully."
