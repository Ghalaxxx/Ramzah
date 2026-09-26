$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$backend = Join-Path $root 'Ramzah_Prototype'
$frontend = Join-Path $root 'Ramzah_Interface_Clean'
$venvPython = Join-Path $backend '.venv\Scripts\python.exe'
$pnpm = Join-Path $env:APPDATA 'npm\pnpm.cmd'

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python is not installed or is not available on PATH.'
}
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw 'Node.js is not installed or is not available on PATH.'
}
if (-not (Test-Path -LiteralPath $pnpm)) {
    & (Join-Path $env:ProgramFiles 'nodejs\npm.cmd') install --global pnpm
}
if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv (Join-Path $backend '.venv')
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $backend 'requirements.txt')

Push-Location $frontend
try { & $pnpm install --frozen-lockfile } finally { Pop-Location }

Write-Host 'Ramzah dependencies are installed.' -ForegroundColor Green
