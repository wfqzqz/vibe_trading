#Requires -Version 5.1
<#
.SYNOPSIS
  Windows 本地开发一键启动（等价于 Linux/macOS 的 scripts/dev）。

.DESCRIPTION
  up / stop / restart / status / logs / urls / open
  后台启动 backend(FastAPI, 默认 8899) + frontend(Vite, 默认 5899)。
  本地路径的 factor-runtime 优雅降级（无 py-alpha-lib 时提示需 Docker）。

.EXAMPLE
  .\scripts\dev.ps1 up
  .\scripts\dev.ps1 status
  .\scripts\dev.ps1 stop
#>
param(
    [Parameter(Position = 0)]
    [string]$Command = ""
)

$ErrorActionPreference = "Stop"

# Resolve repo root (scripts/..), state/log/pid dirs.
$Root = Split-Path -Parent $PSScriptRoot
$StateDir = if ($env:VIBE_DEV_STATE_DIR) { $env:VIBE_DEV_STATE_DIR } else { Join-Path $Root ".vibe-dev" }
$LogDir = Join-Path $StateDir "logs"
$PidDir = Join-Path $StateDir "pids"

$BackendHost = if ($env:VIBE_BACKEND_HOST) { $env:VIBE_BACKEND_HOST } else { "127.0.0.1" }
$BackendPort = if ($env:VIBE_BACKEND_PORT) { $env:VIBE_BACKEND_PORT } else { "8899" }
$FrontendHost = if ($env:VIBE_FRONTEND_HOST) { $env:VIBE_FRONTEND_HOST } else { "127.0.0.1" }
$FrontendPort = if ($env:VIBE_FRONTEND_PORT) { $env:VIBE_FRONTEND_PORT } else { "5899" }

function Get-Python {
    if ($env:PYTHON) { return $env:PYTHON }
    $candidates = @(
        (Join-Path $Root ".venv\Scripts\python.exe"),
        (Join-Path $Root "agent\.venv\Scripts\python.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return "python"
}

function Usage {
    @"
Usage: .\scripts\dev.ps1 <command>

Commands:
  up                 Start backend and frontend dev servers
  stop               Stop dev servers started by this script
  restart [service]  Restart backend, frontend, or all services
  status             Show process status and URLs
  logs [service]     Tail logs for backend, frontend, or all
  open               Open the frontend in the default browser
  urls               Print local dev URLs

Environment:
  VIBE_BACKEND_PORT   Backend port (default: 8899)
  VIBE_FRONTEND_PORT  Frontend port (default: 5899)
  PYTHON              Python binary (default: local .venv, then python)
"@
}

function Get-PidFile([string]$Service) { return Join-Path $PidDir "$Service.pid" }
function Get-LogFile([string]$Service) { return Join-Path $LogDir "$Service.log" }

function Test-Running([string]$Service) {
    $file = Get-PidFile $Service
    if (-not (Test-Path $file)) { return $false }
    $pidValue = (Get-Content $file -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $pidValue) { return $false }
    try { Get-Process -Id ([int]$pidValue) -ErrorAction Stop | Out-Null; return $true }
    catch { return $false }
}

function Test-Url([string]$Url) {
    try {
        $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        return ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500)
    }
    catch { return $false }
}

function Wait-Url([string]$Service, [string]$Url, [int]$Attempts = 30) {
    Write-Host -NoNewline "waiting for $Service at $Url"
    for ($i = 1; $i -le $Attempts; $i++) {
        if (Test-Url $Url) { Write-Host " ready"; return $true }
        if (-not (Test-Running $Service)) {
            Write-Host ""
            Write-Host "$Service exited early; see $(Get-LogFile $Service)" -ForegroundColor Red
            return $false
        }
        Write-Host -NoNewline "."
        Start-Sleep -Seconds 1
    }
    Write-Host ""
    Write-Host "$Service not ready after ${Attempts}s; see $(Get-LogFile $Service)" -ForegroundColor Red
    return $false
}

function Get-ServiceUrl([string]$Service) {
    switch ($Service) {
        "backend"   { return "http://127.0.0.1:$BackendPort/health" }
        "frontend"  { return "http://127.0.0.1:$FrontendPort" }
        default     { return $null }
    }
}

function Start-Backend {
    if (-not (Test-Running "backend") -and (Test-Url (Get-ServiceUrl "backend"))) {
        Remove-Item (Get-PidFile "backend") -ErrorAction SilentlyContinue
        Write-Host "backend already reachable at $(Get-ServiceUrl "backend") (external)"
        return
    }
    if (Test-Running "backend") {
        Write-Host "backend already running (pid $(Get-Content (Get-PidFile 'backend')))"
        return
    }

    $python = Get-Python
    $log = Get-LogFile "backend"
    $errlog = "$log.err"
    $pidFile = Get-PidFile "backend"

    New-Item -ItemType Directory -Force -Path $LogDir, $PidDir | Out-Null
    Write-Host "starting backend..."
    $code = 'import cli, sys; raise SystemExit(cli.main(sys.argv[1:]))'
    $env:PYTHONPATH = Join-Path $Root "agent"
    $proc = Start-Process -FilePath $python `
        -ArgumentList @('-c', $code, 'serve', '--host', $BackendHost, '--port', $BackendPort) `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $log `
        -RedirectStandardError $errlog `
        -PassThru `
        -WindowStyle Hidden
    Set-Content -Path $pidFile -Value $proc.Id
    Write-Host "backend pid $($proc.Id), log $log"
}

function Start-Frontend {
    if (-not (Test-Running "frontend") -and (Test-Url (Get-ServiceUrl "frontend"))) {
        Remove-Item (Get-PidFile "frontend") -ErrorAction SilentlyContinue
        Write-Host "frontend already reachable at $(Get-ServiceUrl "frontend") (external)"
        return
    }
    if (Test-Running "frontend") {
        Write-Host "frontend already running (pid $(Get-Content (Get-PidFile 'frontend')))"
        return
    }

    $frontendDir = Join-Path $Root "frontend"
    if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
        Write-Host "frontend: npm install (first run)..."
        Push-Location $frontendDir
        try { & npm install; if ($LASTEXITCODE -ne 0) { throw "npm install failed" } }
        finally { Pop-Location }
    }

    $log = Get-LogFile "frontend"
    $errlog = "$log.err"
    $pidFile = Get-PidFile "frontend"

    New-Item -ItemType Directory -Force -Path $LogDir, $PidDir | Out-Null
    Write-Host "starting frontend..."
    $env:VITE_API_URL = "http://127.0.0.1:$BackendPort"
    $proc = Start-Process -FilePath "cmd.exe" `
        -ArgumentList @('/c', "npm run dev -- --host $FrontendHost --port $FrontendPort") `
        -WorkingDirectory $frontendDir `
        -RedirectStandardOutput $log `
        -RedirectStandardError $errlog `
        -PassThru `
        -WindowStyle Hidden
    Set-Content -Path $pidFile -Value $proc.Id
    Write-Host "frontend pid $($proc.Id), log $log"
}

function Stop-Service([string]$Service) {
    $pidFile = Get-PidFile $Service
    if (-not (Test-Path $pidFile)) {
        Write-Host "$Service not started by scripts\dev.ps1"
        return
    }
    $pidValue = Get-Content $pidFile -ErrorAction SilentlyContinue
    if ($pidValue -and (Test-Running $Service)) {
        Write-Host "stopping $Service (pid $pidValue)..."
        Stop-Process -Id ([int]$pidValue) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

function Show-Urls {
    @"
Frontend: http://127.0.0.1:$FrontendPort
Backend:  http://127.0.0.1:$BackendPort
API docs: http://127.0.0.1:$BackendPort/docs
"@ | Write-Host
}

function Cmd-Up {
    Start-Backend
    Start-Frontend
    $ok = Wait-Url "backend" (Get-ServiceUrl "backend") 30
    if ($ok) { $ok = Wait-Url "frontend" (Get-ServiceUrl "frontend") 30 }
    if (-not $ok) { Cmd-Stop; exit 1 }
    Show-Urls
}

function Cmd-Stop {
    Stop-Service "frontend"
    Stop-Service "backend"
}

function Cmd-Restart([string]$Service) {
    if (-not $Service) { $Service = "all" }
    switch ($Service) {
        "backend"   { Stop-Service "backend";   Start-Backend;   Wait-Url "backend" (Get-ServiceUrl "backend") 30 | Out-Null }
        "frontend"  { Stop-Service "frontend";  Start-Frontend;  Wait-Url "frontend" (Get-ServiceUrl "frontend") 30 | Out-Null }
        "all"       { Cmd-Stop; Cmd-Up }
        default     { Write-Host "unknown service: $Service"; exit 2 }
    }
}

function Cmd-Status {
    foreach ($svc in @("backend", "frontend")) {
        if (Test-Running $svc) {
            Write-Host ("{0,-8} running pid={1} log={2}" -f $svc, (Get-Content (Get-PidFile $svc)), (Get-LogFile $svc))
        }
        elseif (Test-Url (Get-ServiceUrl $svc)) {
            Write-Host ("{0,-8} reachable external url={1}" -f $svc, (Get-ServiceUrl $svc))
        }
        else {
            Write-Host ("{0,-8} stopped" -f $svc)
        }
    }
    Show-Urls
}

function Cmd-Logs([string]$Service) {
    if (-not $Service) { $Service = "all" }
    $targets = if ($Service -in @("backend", "frontend")) { @($Service) } elseif ($Service -eq "all") { @("backend", "frontend") } else { @() }
    if (-not $targets) { Write-Host "unknown service: $Service"; exit 2 }
    foreach ($t in $targets) {
        $log = Get-LogFile $t
        if (-not (Test-Path $log)) { New-Item -ItemType File -Path $log -Force | Out-Null }
        if (Test-Path "$log.err") { Write-Host "--- $t stderr (last 40) ---"; Get-Content "$log.err" -Tail 40 -ErrorAction SilentlyContinue }
    }
    $paths = $targets | ForEach-Object { Get-LogFile $_ }
    Get-Content -Path $paths -Wait -Tail 40
}

function Cmd-Open {
    Start-Process "http://127.0.0.1:$FrontendPort"
}

$RestArg = if ($args.Count -gt 0) { [string]$args[0] } else { "" }
switch ($Command) {
    "up"       { Cmd-Up }
    "stop"     { Cmd-Stop }
    "restart"  { Cmd-Restart $RestArg }
    "status"   { Cmd-Status }
    "logs"     { Cmd-Logs $RestArg }
    "open"     { Cmd-Open }
    "urls"     { Show-Urls }
    { $_ -in @("", "-h", "--help", "help") } { Usage }
    default    { Write-Host "unknown command: $Command`n"; Usage; exit 2 }
}
