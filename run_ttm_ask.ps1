$ErrorActionPreference = 'Stop'

$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$htmlFile = Join-Path $baseDir 'ttm_ask.html'
$launcherPy = Join-Path $baseDir 'launcher.py'
$appPy = Join-Path $baseDir 'app.py'
$setupScript = Join-Path $baseDir 'setup_windows.ps1'
$logDir = Join-Path $baseDir 'logs'
$runLog = Join-Path $logDir 'run.log'

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Write-RunLog {
    param([Parameter(Mandatory = $true)][string]$Message)
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $runLog -Value "[$timestamp] $Message"
}

function Get-AppPort {
    $configuredPort = [string]$env:TTM_ASK_PORT
    if ([string]::IsNullOrWhiteSpace($configuredPort)) {
        return 5000
    }

    $parsedPort = 0
    if ([int]::TryParse($configuredPort, [ref]$parsedPort) -and $parsedPort -gt 0 -and $parsedPort -le 65535) {
        return $parsedPort
    }

    Write-RunLog "Invalid TTM_ASK_PORT '$configuredPort'. Falling back to 5000."
    return 5000
}

function Resolve-VenvPython {
    $candidates = @(
        (Join-Path $baseDir '.venv\Scripts\python.exe'),
        (Join-Path $baseDir '.venv-1\Scripts\python.exe'),
        (Join-Path $baseDir 'venv\Scripts\python.exe')
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Test-Url {
    param([Parameter(Mandatory = $true)][string]$Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch {
        return $false
    }
}

function Wait-ForLauncher {
    for ($i = 0; $i -lt 15; $i++) {
        if (Test-Url 'http://localhost:5001/status') {
            return $true
        }
        Start-Sleep -Milliseconds 800
    }
    return $false
}

function Wait-ForBackend {
    for ($i = 0; $i -lt 120; $i++) {
        if (Test-Url $healthUrl) {
            return $true
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Start-BackendIfNeeded {
    $startedLocally = $false

    if (Test-Url $healthUrl) {
        Write-RunLog 'Backend already online.'
        return 'online'
    }

    if (-not (Test-Url 'http://localhost:5001/status')) {
        Write-RunLog 'Launcher not running. Starting launcher process.'
        Start-Process -FilePath $venvPython -ArgumentList @($launcherPy) -WorkingDirectory $baseDir -WindowStyle Hidden | Out-Null
        $startedLocally = $true
        if (-not (Wait-ForLauncher)) {
            Write-RunLog 'Launcher failed to come online. Starting backend directly.'
            Start-Process -FilePath $venvPython -ArgumentList @($appPy) -WorkingDirectory $baseDir -WindowStyle Hidden | Out-Null
            $startedLocally = $true
            if (Wait-ForBackend) {
                Write-RunLog 'Backend started directly without launcher.'
                return 'online'
            }
            Write-RunLog 'Direct backend start is still warming up.'
            return 'warming'
        }
    }

    try {
        Invoke-WebRequest -Uri 'http://localhost:5001/start' -UseBasicParsing -TimeoutSec 5 | Out-Null
        $startedLocally = $true
    } catch {
        Write-RunLog 'Launcher /start request failed; continuing with backend checks.'
    }

    if (Wait-ForBackend) {
        Write-RunLog 'Backend is online.'
        return 'online'
    }

    if ($startedLocally) {
        Write-RunLog 'Backend did not come online yet; continuing while it warms up.'
        return 'warming'
    }

    Write-RunLog 'Backend is offline and launcher start was unavailable.'
    return 'failed'
}

$appPort = Get-AppPort
$appUrl = "http://localhost:$appPort/"
$healthUrl = "http://localhost:$appPort/health"

$venvPython = Resolve-VenvPython
if (-not $venvPython) {
    Write-Host 'No virtual environment found. Running first-time setup ...' -ForegroundColor Yellow
    Write-RunLog 'No virtual environment found. Running setup.'
    & powershell -ExecutionPolicy Bypass -File $setupScript
    if ($LASTEXITCODE -ne 0) {
        throw 'Initial setup failed.'
    }
    $venvPython = Resolve-VenvPython
}

$backendStartupState = Start-BackendIfNeeded
if ($backendStartupState -eq 'failed') {
    Write-Warning 'Backend did not start. Running setup repair once and retrying.'
    Write-RunLog 'Backend offline after first attempt. Running setup repair.'
    & powershell -ExecutionPolicy Bypass -File $setupScript
    if ($LASTEXITCODE -eq 0) {
        $venvPython = Resolve-VenvPython
        $backendStartupState = Start-BackendIfNeeded
        if ($backendStartupState -eq 'failed') {
            Write-Warning 'Backend is still offline. Check logs\\setup.log and logs\\run.log.'
            Write-RunLog 'Backend still offline after repair.'
        }
    } else {
        Write-Warning 'Setup repair failed. Check logs\\setup.log for details.'
        Write-RunLog 'Setup repair failed.'
    }
}

if ($backendStartupState -eq 'online') {
    Start-Process -FilePath $appUrl | Out-Null
    Write-RunLog "Local app launched at $appUrl"
} else {
    Start-Process -FilePath $appUrl | Out-Null
    Write-RunLog "Launched $appUrl while backend continues to warm up."
}
