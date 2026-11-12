# Start Ollama for Valta. Both modes share one model folder.
#
# -Mode separate (default): two servers.
#   Port 11434: Vulkan on the RX 6600 only (teacher and embeddings). CUDA is hidden, and Vulkan is limited
#   to device 0, the RX 6600; Vulkan also sees the GTX 1660, and would put a model too big for the RX 6600
#   partly on it, under a training run. Check data\logs\ollama-rx6600.err.log for "using device Vulkan0".
#   Port 11435: CUDA on the GTX 1660 (personal model). Vulkan is off.
#
# -Mode combined: one server on port 11434 that sees both cards through Vulkan and
#   spreads a model across them (about 14 GB together), for a teacher too big for
#   either card alone, such as qwen2.5-coder:14b. Port 11435 stays down, so there is
#   no personal model, and `valta train`, `valta practice` and `valta eval` without
#   --teacher will say the GTX 1660 server is down. Training uses the 1660 directly
#   and cannot run while this server holds it.
#
# Switch modes with scripts\stop-ollama.ps1 first.
# Quit the Ollama tray app first. It binds 11434 and will ignore these settings.
#
# Each server writes data\logs\ollama-<name>.err.log. If one does not come up,
# this script says so and prints the end of that log, which holds the reason.

param(
    [ValidateSet("separate", "combined")]
    [string]$Mode = "separate"
)

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

if ($Mode -eq "combined") {
    if ((Test-PortOpen 11434) -or (Test-PortOpen 11435)) {
        Write-Host "Ollama is already running on 11434 or 11435. Run scripts\stop-ollama.ps1 first, then this again." -ForegroundColor Yellow
        return
    }
    # CUDA is hidden so the GTX 1660 is reached through Vulkan like the RX 6600; one backend can split a
    # model across both. GGML_VK_VISIBLE_DEVICES is cleared so Vulkan sees every card, and
    # OLLAMA_SCHED_SPREAD spreads a model across all of them even when one card could hold it.
    Start-OllamaServer "combined" 11434 @{
        CUDA_VISIBLE_DEVICES = "-1"
        OLLAMA_VULKAN = "1"
        GGML_VK_VISIBLE_DEVICES = ""
        OLLAMA_SCHED_SPREAD = "1"
    }
    Write-Host "Combined mode: one server on 11434 using both cards. No personal model on 11435."
    Write-Host "Check the split with: ollama ps   (and data\logs\ollama-combined.err.log for per-card layers)"
    Write-Host "Model folder: $Models"
    return
}

if (Test-PortOpen 11434) {
    Write-Host "Port 11434 is already in use. Quit Ollama from the system tray, then run this script again so the RX 6600 server can bind it."
} else {
    Start-OllamaServer "rx6600" 11434 @{
        CUDA_VISIBLE_DEVICES = "-1"
        OLLAMA_VULKAN = "1"
        GGML_VK_VISIBLE_DEVICES = "0"
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
