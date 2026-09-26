$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$backend = Join-Path $root 'Ramzah_Prototype'
$frontend = Join-Path $root 'Ramzah_Interface_Clean'
$python = Join-Path $backend '.venv\Scripts\python.exe'
$pnpm = Join-Path $env:APPDATA 'npm\pnpm.cmd'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Dependencies are not installed. Run setup_ramzah.cmd first.'
}
if (-not (Test-Path -LiteralPath $pnpm)) {
    throw 'pnpm is not installed. Run setup_ramzah.cmd first.'
}

$existing = Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue
$backendProcess = $null

try {
    if (-not $existing) {
        $backendProcess = Start-Process -FilePath $python `
            -ArgumentList '-m','uvicorn','ramzah.api:app','--host','127.0.0.1','--port','8000' `
            -WorkingDirectory $backend -WindowStyle Hidden -PassThru

        $ready = $false
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            Start-Sleep -Milliseconds 500
            try {
                $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 2
                if ($health.status -eq 'ready') { $ready = $true; break }
            } catch {}
        }
        if (-not $ready) { throw 'The Ramzah backend did not become ready on port 8000.' }
    }

    Write-Host 'Ramzah is ready at http://localhost:3000' -ForegroundColor Green
    Push-Location $frontend
    try { & $pnpm dev } finally { Pop-Location }
}
finally {
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force
    }
}
