param([switch]$IncludeLlm)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$resultsPath = Join-Path $repoRoot ".work\evaluation-$runStamp"
New-Item -ItemType Directory -Path $resultsPath | Out-Null
$offline = @('--rm', '--no-deps', '-e', 'LLM_ENABLED=false', '-e', 'LLM_EXTRACTION_ENABLED=false', '-e', 'LLM_SELECTION_ENABLED=false', '-e', 'SOLUTION_CACHE_ENABLED=false', '-e', 'COMPUTATION_CACHE_SECONDS=0', '-e', 'LLM_CACHE_SECONDS=0', '-e', 'CONVERSATION_STORAGE_ENABLED=false', '-v', "${resultsPath}:/results")
docker compose exec -T api python -m scripts clear-caches
if ($LASTEXITCODE -ne 0) { throw 'Cache reset failed.' }
docker compose restart api
if ($LASTEXITCODE -ne 0) { throw 'API restart failed.' }
$ready = $false
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/v1/ready' -TimeoutSec 5
        $ready = $true
        break
    } catch { Start-Sleep -Seconds 2 }
}
if (-not $ready) { throw 'API did not become ready. Inspect docker compose logs api.' }
foreach ($split in @('dev', 'test')) {
    docker compose run @offline api python -m scripts evaluate --split $split --output "/results/$split.json"
    if ($LASTEXITCODE -ne 0) { throw "$split pipeline evaluation failed. Existing reports are preserved." }
}
docker compose run @offline api python -m scripts quality --output /results/quality-local.json
if ($LASTEXITCODE -ne 0) { throw 'Local response-quality replay failed.' }
if ($IncludeLlm) {
    $live = @('--rm', '--no-deps', '-e', 'LLM_ENABLED=true', '-e', 'LLM_EXTRACTION_ENABLED=true', '-e', 'LLM_SELECTION_ENABLED=true', '-e', 'SOLUTION_CACHE_ENABLED=false', '-e', 'COMPUTATION_CACHE_SECONDS=0', '-e', 'LLM_CACHE_SECONDS=0', '-e', 'CONVERSATION_STORAGE_ENABLED=false', '-v', "${resultsPath}:/results")
    $cases = (Get-Content -LiteralPath data/evaluation/customer_challenge_cases_v1.json -Raw -Encoding utf8 | ConvertFrom-Json).cases | Select-Object -First 3
    $caseArgs = @()
    foreach ($case in $cases) { $caseArgs += @('--case', $case.id) }
    docker compose run @live api python -m scripts quality @caseArgs --delay-seconds 60 --output /results/quality-llm.json
    if ($LASTEXITCODE -ne 0) { throw 'Small LLM replay failed; inspect the partial report for quota/provider failures.' }
}
$scores = foreach ($split in @('dev', 'test')) {
    $report = Get-Content -LiteralPath (Join-Path $resultsPath "$split.json") -Raw -Encoding utf8 | ConvertFrom-Json
    $published = $report.classifier_comparisons.PSObject.Properties | Where-Object Name -Like 'published_*' | Select-Object -First 1 -ExpandProperty Value
    [PSCustomObject]@{
        Split = $split
        CategoryAccuracy = $published.accuracy
        MacroF1 = $published.macro_f1
        KBHitAt5 = $report.summary.retrieval.kb_focused.hit_at_5
        CitationValidity = $report.summary.citation_contract_pass_rate
        SeverityCorrect = $report.summary.signals.severity_correct
        SentimentCorrect = $report.summary.signals.sentiment_correct
        SeverityUnknown = $report.summary.signals.severity_unknown
        SentimentUnknown = $report.summary.signals.sentiment_unknown
        ComparisonP50ms = $report.summary.comparison_latency_ms.p50
        ComparisonP95ms = $report.summary.comparison_latency_ms.p95
    }
}
$scores | Format-Table -AutoSize
$scores | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resultsPath 'scores.json') -Encoding utf8
Write-Host "Reports saved in $resultsPath"
Write-Host 'Quality reports contain evidence and empty human-rating fields. Review plans yourself; citation validity is not semantic correctness.'
