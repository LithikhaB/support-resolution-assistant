param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [switch]$NoReload,
    [switch]$Check
)

Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    Write-Error 'Project environment missing. Run: python -m venv .venv, then .\.venv\Scripts\python.exe -m pip install -r requirements.txt'
    exit 1
}
& $projectPython -c 'import uvicorn, psycopg; import app.main'
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Project dependencies are missing or the application cannot load. Run: .\.venv\Scripts\python.exe -m pip install -r requirements.txt'
    exit 1
}
if ($Check) {
    Write-Output 'Project environment and application imports are ready.'
    exit 0
}
$serverArguments = @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$Port")
if (-not $NoReload) { $serverArguments += '--reload' }
& $projectPython @serverArguments
exit $LASTEXITCODE
