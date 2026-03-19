param(
    [switch]$Install,
    [switch]$Test,
    [switch]$Demo
)

$ErrorActionPreference = 'Stop'

if (-not ($Install -or $Test -or $Demo)) {
    Write-Host 'Usage: ./scripts/dev.ps1 -Install | -Test | -Demo'
    exit 1
}

if ($Install) {
    python -m pip install -r requirements.txt
}

if ($Test) {
    python -m pytest
}

if ($Demo) {
    python ai_text_generator.py detect "Hello world"
    python -m ai_tel.cli analyze "This project is now structured like a package."
    python -m ai_tel.cli generate "machine learning" --language english --length 80
}
