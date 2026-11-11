# Start two Ollama servers that share one model folder.
# Port 11434: Vulkan on the RX 6600 (teacher and embeddings). CUDA is hidden.
# Port 11435: CUDA on the GTX 1660 (personal model). Vulkan is off.
# Quit the Ollama tray app first. It binds 11434 and will ignore these settings.
#
# Each server writes data\logs\ollama-<name>.err.log. If one does not come up,
# this script says so and prints the end of that log, which holds the reason.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Models = Join-Path $env:USERPROFILE ".ollama\models"
$Logs = Join-Path $Root "data\logs"
New-Item -ItemType Directory -Force -Path $Models | Out-Null
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

function Test-PortOpen([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $Port)
        return $true
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Start-OllamaServer([string]$Name, [int]$Port, [hashtable]$Extra) {
    $settings = @{ OLLAMA_HOST = "127.0.0.1:$Port"; OLLAMA_MODELS = $Models }
    foreach ($key in $Extra.Keys) {
        $settings[$key] = $Extra[$key]
    }
    $saved = @{}
    foreach ($key in $settings.Keys) {
        $saved[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, $settings[$key], "Process")
    }
    $out = Join-Path $Logs "ollama-$Name.out.log"
    $err = Join-Path $Logs "ollama-$Name.err.log"
    $process = $null
    try {
        $process = Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $out -RedirectStandardError $err
    } catch {
        Write-Host "Could not start ollama: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Is Ollama installed and on PATH? Try: ollama --version"
        return
    } finally {
        foreach ($key in $saved.Keys) {
            [Environment]::SetEnvironmentVariable($key, $saved[$key], "Process")
        }
    }
    for ($i = 0; $i -lt 40; $i++) {
        if (Test-PortOpen $Port) {
            Write-Host "Started $Name (pid $($process.Id)) on port $Port"
            return
        }
        if ($process.HasExited) { break }
        Start-Sleep -Milliseconds 500
    }
    Write-Host "The $Name server did not come up on port $Port." -ForegroundColor Red
    if ($process.HasExited) {
        Write-Host "It exited with code $($process.ExitCode)."
    }
    if (Test-Path $err) {
        Write-Host "Last lines of ${err}:"
        Get-Content $err -Tail 15
    }
}

if (Test-PortOpen 11434) {
    Write-Host "Port 11434 is already in use. Quit Ollama from the system tray, then run this script again so the RX 6600 server can bind it."
} else {
    Start-OllamaServer "rx6600" 11434 @{
        CUDA_VISIBLE_DEVICES = "-1"
        OLLAMA_VULKAN = "1"
    }
}

if (Test-PortOpen 11435) {
    Write-Host "Port 11435 is already open."
} else {
    Start-OllamaServer "gtx1660" 11435 @{
        CUDA_VISIBLE_DEVICES = "0"
        OLLAMA_VULKAN = "0"
        GGML_VK_VISIBLE_DEVICES = "-1"
    }
}

Write-Host "Model folder: $Models"
Write-Host "Then run: valta models pull"
