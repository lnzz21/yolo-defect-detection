$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Create the project .venv and install its dependencies first. See WORKSTATION.md.'
}

Push-Location -LiteralPath $PSScriptRoot
try {
    & $python main.py
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
