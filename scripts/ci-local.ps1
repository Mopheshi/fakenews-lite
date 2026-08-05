<#
.SYNOPSIS
    Runs the CI pipeline locally, mirroring .github/workflows/ci.yml.

.DESCRIPTION
    Executes the same steps GitHub Actions would: install, fetch NLTK resources,
    lint, test. Useful when Actions is unavailable (billing, outage, fork
    restrictions) or to check a change before pushing.

    Unlike CI, this exercises only the interpreter you point it at, not the full
    3.11/3.12/3.13 matrix.

.PARAMETER Python
    Interpreter to use. Defaults to the project virtual environment.

.PARAMETER SkipInstall
    Skip the dependency install when the environment is already provisioned.

.EXAMPLE
    ./scripts/ci-local.ps1
    ./scripts/ci-local.ps1 -SkipInstall
#>
[CmdletBinding()]
param(
    [string]$Python,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'

# $PSScriptRoot is not reliably populated inside param() defaults, so the repo
# root and the default interpreter are both resolved here instead.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')

if (-not $Python) {
    $Python = Join-Path $repoRoot '.venv\Scripts\python.exe'
}

Push-Location $repoRoot

$failed = @()

function Invoke-Step {
    param([string]$Name, [scriptblock]$Action)

    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    try {
        & $Action
        # Native executables report failure through $LASTEXITCODE, not exceptions.
        if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
        Write-Host "PASS  $Name" -ForegroundColor Green
    }
    catch {
        Write-Host "FAIL  $Name -- $_" -ForegroundColor Red
        $script:failed += $Name
    }
}

if (-not (Test-Path $Python)) {
    Write-Host "Interpreter not found: $Python" -ForegroundColor Red
    Pop-Location
    exit 1
}

$version = & $Python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Write-Host "Repository : $repoRoot"
Write-Host "Interpreter: $Python ($version)"

if (-not $SkipInstall) {
    Invoke-Step "Install" {
        & $Python -m pip install --upgrade pip --quiet
        & $Python -m pip install -e ".[app,bench,dev]" --quiet
    }
}

Invoke-Step "NLTK resources" {
    & $Python -c "from fakenews.nltk_resources import ensure_nltk_resources; ensure_nltk_resources()"
}

Invoke-Step "Lint" { & $Python -m ruff check . }

Invoke-Step "Test" { & $Python -m pytest -q }

Write-Host ""
if ($failed.Count -gt 0) {
    Write-Host "CI FAILED: $($failed -join ', ')" -ForegroundColor Red
    Pop-Location
    exit 1
}

Write-Host "CI PASSED (Python $version)" -ForegroundColor Green
Write-Host "Note: CI covers 3.11, 3.12 and 3.13; this run exercised $version only." -ForegroundColor Yellow
Pop-Location
