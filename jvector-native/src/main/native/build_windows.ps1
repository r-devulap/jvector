# Copyright DataStax, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Windows build script for jvector_simd_kernels.
# Run from a Visual Studio Developer PowerShell (or any PowerShell where
# cl.exe is on PATH) so that Meson auto-detects MSVC.
# To use clang++ instead, set $env:CXX = 'clang++' before running.
#
# Usage:
#   .\build_windows.ps1 [release|debug|debugoptimized]
#
# The buildtype parameter is optional and defaults to 'release'.

param(
    [ValidateSet('release', 'debug', 'debugoptimized')]
    [string]$BuildType = 'release'
)

$ErrorActionPreference = 'Stop'

Write-Host "BUILDTYPE=$BuildType"

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

# Google Highway submodule
$HighwayHeader = Join-Path $PSScriptRoot 'third_party\highway\hwy\highway.h'
if (-not (Test-Path $HighwayHeader)) {
    Write-Error @"
ERROR: Google Highway submodule not found at third_party\highway.
       Run the following command from the repository root to fix this:

         git submodule update --init
"@
    exit 1
}

# meson
if (-not (Get-Command meson -ErrorAction SilentlyContinue)) {
    Write-Error "ERROR: meson is not installed or not on PATH.`n       Install it with: pip install meson"
    exit 2
}

# ninja
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    Write-Error "ERROR: ninja is not installed or not on PATH.`n       Install it with: pip install ninja"
    exit 2
}

# ---------------------------------------------------------------------------
# Compiler check
# ---------------------------------------------------------------------------
# Meson auto-detects cl.exe when inside a VS Developer shell.
# If $env:CXX is set it takes precedence (e.g. $env:CXX = 'clang++').
$MinClangVersion = 14

if (Get-Command cl -ErrorAction SilentlyContinue) {
    Write-Host 'Compiler: MSVC (cl.exe)'
} elseif (Get-Command clang++ -ErrorAction SilentlyContinue) {
    $clangVerLine = & clang++ --version 2>&1 | Select-String 'clang version (\d+)' | Select-Object -First 1
    if ($clangVerLine -match 'clang version (\d+)') {
        $clangMajor = [int]$Matches[1]
        if ($clangMajor -lt $MinClangVersion) {
            Write-Warning "clang++ version $clangMajor is too old. Please upgrade to clang++ $MinClangVersion or newer for full tier support."
        }
        Write-Host "Compiler: clang++ $clangMajor"
    } else {
        Write-Warning 'Could not determine clang++ version. Proceeding anyway.'
        Write-Host 'Compiler: clang++'
    }
} else {
    Write-Error @"
ERROR: No supported C++ compiler found on PATH.
       Install one of:
         - Visual Studio 2019+ (with C++ workload) and run from a VS Developer PowerShell
         - LLVM for Windows (https://releases.llvm.org/) with clang++ >= $MinClangVersion
"@
    exit 2
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

$BuildDir  = Join-Path $PSScriptRoot '..\..\..\target\meson-build'
$ResourcesDir = Join-Path $PSScriptRoot '..\resources'
$LibOut    = Join-Path $ResourcesDir 'jvector.dll'

New-Item -ItemType Directory -Force -Path $ResourcesDir | Out-Null

if (Test-Path $LibOut) { Remove-Item $LibOut -Force }

# Change to the native source directory so meson picks up meson.build
Push-Location $PSScriptRoot
try {
    & meson setup $BuildDir --wipe "--buildtype=$BuildType"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & meson compile -C $BuildDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}

$DllFile = Get-ChildItem -Path $BuildDir -Depth 0 -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^jvector.*\.dll$' } |
    Select-Object -First 1
if (-not $DllFile) {
    Write-Error "ERROR: jvector.dll not found in $BuildDir after build."
    exit 1
}
Copy-Item $DllFile.FullName $LibOut
Write-Host "Copied $($DllFile.FullName) -> $LibOut"

# ---------------------------------------------------------------------------
# jextract (optional — only needed when jvector_simd.h changes)
# ---------------------------------------------------------------------------

if (-not (Get-Command jextract -ErrorAction SilentlyContinue)) {
    Write-Warning 'jextract could not be found, please install it if you need to update bindings.'
    exit 0
}

$JavaOut   = Join-Path $PSScriptRoot '..\java'
$HeaderFile = Join-Path $PSScriptRoot 'jvector_simd.h'

& jextract `
    --output $JavaOut `
    -t io.github.jbellis.jvector.vector.cnative `
    -I $PSScriptRoot `
    --header-class-name NativeSimdOps `
    $HeaderFile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Patch generated file: add Linker.Option.critical(true) to every downcall.
$GeneratedFile = Join-Path $JavaOut 'io\github\jbellis\jvector\vector\cnative\NativeSimdOps.java'
(Get-Content $GeneratedFile) -replace 'DESC\)', 'DESC, Linker.Option.critical(true))' |
    Set-Content $GeneratedFile
Write-Host "Patched $GeneratedFile"
