<#
.SYNOPSIS
    Install and verify LSTP, then generate the SLAI benchmark report.
.EXAMPLE
    .\benchmarks\lstp\run_benchmark.ps1 -LSTPPath 'G:\GAIA\AI\LSTP'
.EXAMPLE
    .\benchmarks\lstp\run_benchmark.ps1 -LSTPPath 'G:\GAIA\AI\LSTP' -Live `
        -ConfirmPricing -ConfirmGroqFreeTier
#>
[CmdletBinding()]
param(
    [string]$LSTPPath = '',
    [ValidateRange(1,5400)][int]$Cases = 1000,
    [int]$Seed = 20261117,
    [string]$RunName = 'paired_1000_v02',
    [switch]$Live,
    [switch]$ConfirmPricing,
    [switch]$ConfirmGroqFreeTier,
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$SLAIRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') {
    throw 'RunName must contain only letters, digits, underscores, or hyphens.'
}
if ([string]::IsNullOrWhiteSpace($LSTPPath)) {
    $LSTPPath = Join-Path $SLAIRoot 'model\LSTP'
}
if (-not (Test-Path -LiteralPath (Join-Path $LSTPPath 'pyproject.toml'))) {
    throw "No initialized LSTP checkout at '$LSTPPath'. Clone it or initialize the SLAI submodule first."
}
$LSTPPath = (Resolve-Path -LiteralPath $LSTPPath).Path
$Output = Join-Path $PSScriptRoot (Join-Path 'results' $RunName)
New-Item -ItemType Directory -Path $Output -Force | Out-Null

$Python = Join-Path $SLAIRoot '.venv-lstp-benchmark\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    $pyLauncher = Get-Command 'py' -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & py -3.12 -m venv (Join-Path $SLAIRoot '.venv-lstp-benchmark')
    }
    else {
        & python -m venv (Join-Path $SLAIRoot '.venv-lstp-benchmark')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python virtual environment.' }
}
if (-not $SkipInstall) {
    & $Python -m pip install -e "${LSTPPath}[dev]"
    if ($LASTEXITCODE -ne 0) { throw 'LSTP development installation failed.' }
}

function Invoke-OfflineCheck {
    param([string]$Name, [string]$Log, [string[]]$Arguments)
    # Windows PowerShell turns a native program's stderr into NativeCommandError
    # when the caller selected Stop. Preserve the actual process exit code.
    $ErrorActionPreference = 'Continue'
    Write-Host "Checking $Name ..."
    $target = Join-Path $Output $Log
    if ($Arguments[0] -eq 'node') {
        $nodeArgs = $Arguments[1..($Arguments.Length - 1)]
        & node @nodeArgs *> $target
    }
    else {
        & $Python @Arguments *> $target
    }
    $success = ($LASTEXITCODE -eq 0)
    if ($success) { Write-Host "PASS: $Name" -ForegroundColor Green }
    else { Write-Warning "FAILED: $Name (details in $target)" }
    return $success
}

$originalLocation = Get-Location
try {
    Set-Location -LiteralPath $LSTPPath
    $schemaOk = Invoke-OfflineCheck 'LSTP schema' 'schema.log' @('tools/check_schema_contract.py')
    $testsOk = Invoke-OfflineCheck 'Core LSTP Python conformance' 'pytest.log' @(
        '-m', 'pytest', '-q',
        'tests/test_conformance_vectors.py',
        'tests/test_canonical_serializer.py',
        'tests/test_canonical_lattice.py',
        'tests/test_semantic_validator.py'
    )
    $interopOk = Invoke-OfflineCheck 'Independent JavaScript conformance' 'interop_js.log' @(
        'node', 'interop/js/verify.mjs'
    )
    $releaseOk = Invoke-OfflineCheck 'Full engineering release checks' 'release_checks.log' @(
        'tools/run_release_checks.py'
    )
}
finally {
    Set-Location -LiteralPath $originalLocation
}
try {
    Set-Location -LiteralPath $SLAIRoot
    $adapterOk = Invoke-OfflineCheck 'SLAI adapter smoke test' 'slai_adapter.log' @(
        '-m', 'benchmarks.lstp.smoke_integration'
    )
}
finally {
    Set-Location -LiteralPath $originalLocation
}

$preflightData = @{
    schema = [bool]$schemaOk
    pytest = [bool]$testsOk
    interop_js = [bool]$interopOk
    slai_adapter = [bool]$adapterOk
    release_runner = [bool]$releaseOk
    recorded_at_utc = [DateTime]::UtcNow.ToString('o')
} | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText(
    (Join-Path $Output 'preflight.json'),
    $preflightData,
    ([System.Text.UTF8Encoding]::new($false))
)

if (-not ($schemaOk -and $testsOk -and $interopOk -and $adapterOk)) {
    throw 'Critical offline checks failed. No live API calls will be made.'
}
if (-not $releaseOk) {
    Write-Warning 'Engineering release gate failed. Experimental benchmark may continue because critical protocol tests passed.'
}

if ($Live) {
    if (-not ($ConfirmPricing -and $ConfirmGroqFreeTier)) {
        throw 'Live mode requires -ConfirmPricing and -ConfirmGroqFreeTier.'
    }
    $values = @{}
    $envFile = Join-Path $SLAIRoot '.env'
    if (Test-Path -LiteralPath $envFile) {
        foreach ($line in Get-Content -LiteralPath $envFile) {
            if ($line -match '^\s*(?:export\s+)?(OPENAI_API_KEY|GROQ_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*(.*)$') {
                $name = $Matches[1]
                $value = $Matches[2].Trim()
                if ($value.StartsWith('"') -and $value.EndsWith('"') -and $value.Length -ge 2) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
                elseif ($value.StartsWith("'") -and $value.EndsWith("'") -and $value.Length -ge 2) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
                else {
                    $value = ($value -replace '\s+#.*$', '').Trim()
                }
                if (-not [string]::IsNullOrWhiteSpace($value)) { $values[$name] = $value }
            }
        }
        if (-not $values.ContainsKey('GEMINI_API_KEY') -and $values.ContainsKey('GOOGLE_API_KEY')) {
            $values['GEMINI_API_KEY'] = $values['GOOGLE_API_KEY']
        }
    }
    foreach ($keyName in @('OPENAI_API_KEY', 'GROQ_API_KEY', 'GEMINI_API_KEY')) {
        $current = [Environment]::GetEnvironmentVariable($keyName, 'Process')
        if ([string]::IsNullOrWhiteSpace($current) -and $values.ContainsKey($keyName)) {
            [Environment]::SetEnvironmentVariable($keyName, $values[$keyName], 'Process')
            $current = $values[$keyName]
            Write-Host "Loaded $keyName from local .env"
        }
        if ([string]::IsNullOrWhiteSpace($current)) {
            $secret = Read-Host -Prompt "Enter $keyName (hidden; not saved)" -AsSecureString
            $networkCredential = [System.Net.NetworkCredential]::new('', $secret)
            if ([string]::IsNullOrWhiteSpace($networkCredential.Password)) {
                throw "Missing $keyName; no API calls will be made."
            }
            [Environment]::SetEnvironmentVariable($keyName, $networkCredential.Password, 'Process')
        }
    }
}

$argv = @(
    '-m', 'benchmarks.lstp.benchmark',
    '--slai-root', $SLAIRoot,
    '--lstp-root', $LSTPPath,
    '--output', $Output,
    '--cases', [string]$Cases,
    '--seed', [string]$Seed
)
if ($Live) { $argv += @('--live', '--confirm-free-groq', '--confirm-pricing') }
Set-Location -LiteralPath $SLAIRoot
try {
    & $Python @argv
    $status = $LASTEXITCODE
}
finally {
    Set-Location -LiteralPath $originalLocation
}
Write-Host "Results: $Output"
Write-Host "Markdown report: $(Join-Path $Output 'final_report.md')"
Write-Host "Machine-readable report: $(Join-Path $Output 'final_report.json')"
if ($status -ne 0) {
    if ($status -eq 2 -and (Test-Path -LiteralPath (Join-Path $Output 'final_report.json'))) {
        Write-Warning 'Provider failure or incomplete data. The partial report is available; inspect its stop_reasons and errors.'
        return
    }
    throw "Benchmark finished without a complete measured report (exit code $status). Run the same command again to resume."
}
