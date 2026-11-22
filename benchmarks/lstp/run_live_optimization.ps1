[CmdletBinding()]
param(
    [ValidateRange(1, 5400)][int]$Cases = 1000,
    [ValidateRange(0.01, 1000)][decimal]$BudgetPerProvider = 5.00,
    [string]$LSTPPath = $env:LSTP_ROOT,
    [string]$SLAIPath = '',
    [string]$PythonPath = '',
    [string]$EnvFile = '',
    [string]$OutputDirectory = '' 
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($SLAIPath)) {
    $SLAIPath = Join-Path $PSScriptRoot '..\..'
}
$SLAI = (Resolve-Path -LiteralPath $SLAIPath).Path
if ([string]::IsNullOrWhiteSpace($LSTPPath)) {
    $ParentRoot = Split-Path -Parent $SLAI
    $Candidates = @(
        (Join-Path $ParentRoot 'LSTP'),
        (Join-Path $ParentRoot 'lstp-optimization'),
        (Join-Path $SLAI 'model\LSTP')
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath (Join-Path $Candidate 'src\lstp\inference.py')) {
            $LSTPPath = $Candidate
            break
        }
    }
}
if ([string]::IsNullOrWhiteSpace($LSTPPath)) {
    throw 'Provide -LSTPPath pointing to the LSTP optimization checkout.'
}
$LSTP = (Resolve-Path -LiteralPath $LSTPPath).Path
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $VenvPython = Join-Path $SLAI '.venv-lstp-benchmark\Scripts\python.exe'
    if (Test-Path -LiteralPath $VenvPython) { $PythonPath = $VenvPython }
    else {
        $PythonCommand = Get-Command python -CommandType Application -ErrorAction SilentlyContinue
        if (-not $PythonCommand) { throw 'Provide -PythonPath for Python 3.11 through 3.14.' }
        $PythonPath = $PythonCommand.Source
    }
}
$Python = (Resolve-Path -LiteralPath $PythonPath).Path
$Preflight = Join-Path $SLAI 'benchmarks\lstp\recorded_verification_2026-11-18.json'
$Pricing = Join-Path $SLAI 'benchmarks\lstp\pricing_2026-11-18.json'
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $SLAI "benchmarks\lstp\results\live_openai_gemini_${Cases}_v1"
}
$Output = [System.IO.Path]::GetFullPath($OutputDirectory)

foreach ($RequiredPath in @($Python, $Preflight, $Pricing)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Required file missing: $RequiredPath"
    }
}

# Reuse recorded verification only when all measured source files still match.
@'
import hashlib, json, pathlib, sys
lstp, slai, receipt = map(pathlib.Path, sys.argv[1:])
record = json.loads(receipt.read_text(encoding="utf-8"))
for name, expected in record["source_sha256"].items():
    source = lstp / "src" / name if name.startswith("lstp/") else slai / "benchmarks/lstp" / name
    data = source.read_bytes()
    if record.get("source_hash_profile") == "utf8-lf":
        data = data.replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit("Verified source changed: " + name)
print("Recorded verification matches; starting real API benchmark.", flush=True)
'@ | & $Python - $LSTP $SLAI $Preflight
if ($LASTEXITCODE -ne 0) { throw 'Source verification failed.' }

# Read only the two provider keys; preserve existing process credentials.
if ([string]::IsNullOrWhiteSpace($EnvFile)) { $EnvFile = Join-Path $SLAI '.env' }
if (Test-Path -LiteralPath $EnvFile) {
    foreach ($Line in Get-Content -LiteralPath $EnvFile) {
        if ($Line -match '^\s*(?:export\s+)?(OPENAI_API_KEY|GEMINI_API_KEY)\s*=\s*(.*)$') {
            $KeyName = $Matches[1]
            $KeyValue = $Matches[2].Trim()
            if ($KeyValue.Length -ge 2 -and $KeyValue[0] -in @([char]34, [char]39)) {
                $Quote = [string]$KeyValue[0]
                $ClosingQuote = $KeyValue.LastIndexOf($Quote)
                if ($ClosingQuote -gt 0) { $KeyValue = $KeyValue.Substring(1, $ClosingQuote - 1) }
            }
            else { $KeyValue = ($KeyValue -replace '\s+#.*$', '').Trim() }
            if (-not [Environment]::GetEnvironmentVariable($KeyName, 'Process')) {
                [Environment]::SetEnvironmentVariable($KeyName, $KeyValue, 'Process')
            }
            $KeyValue = $null
        }
    }
}
foreach ($KeyName in @('OPENAI_API_KEY', 'GEMINI_API_KEY')) {
    if (-not [Environment]::GetEnvironmentVariable($KeyName, 'Process')) {
        $Secret = Read-Host "Enter $KeyName" -AsSecureString
        $KeyValue = [System.Net.NetworkCredential]::new('', $Secret).Password.Trim()
        if (-not $KeyValue) { throw "$KeyName is empty." }
        [Environment]::SetEnvironmentVariable($KeyName, $KeyValue, 'Process')
        $KeyValue = $null
    }
}

$PreviousPythonPath = $env:PYTHONPATH
Push-Location -LiteralPath $SLAI
try {
    $env:PYTHONPATH = "$(Join-Path $LSTP 'src');$SLAI"
    $BudgetText = $BudgetPerProvider.ToString([Globalization.CultureInfo]::InvariantCulture)
    Write-Host "Live requests planned: $($Cases * 9 * 2). Budget guard: USD $BudgetText per provider."
    & $Python -u -m benchmarks.lstp.optimization --live --cases $Cases --seed 20261117 `
        --pricing $Pricing --preflight $Preflight --budget-usd $BudgetText `
        --max-output-tokens 512 --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Live benchmark stopped with an error; inspect its output above.' }
    $ReportPath = Join-Path $Output 'optimization_report.json'
    $Report = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
    $Report.live.metrics.PSObject.Properties | ForEach-Object {
        $Metric = $_.Value
        [pscustomobject]@{
            Condition = $_.Name
            Correct = "$($Metric.successful_tasks)/$($Metric.expected_tasks)"
            Responded = $Metric.completed_tasks
            Tokens = $Metric.known_total_tokens
            USD = $Metric.total_cost
        }
    } | Format-Table -AutoSize
    $Stops = $Report.live.stops | ConvertTo-Json -Compress
    if ($Stops -ne '{}') { Write-Warning "Provider stop reasons: $Stops" }
    Write-Host "Report: $ReportPath"
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}
