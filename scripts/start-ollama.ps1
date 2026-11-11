# Start two Ollama servers that share one model folder.
# Port 11434: Vulkan on the RX 6600 (teacher and embeddings). CUDA is hidden.
# Port 11435: CUDA on the GTX 1660 (personal model). Vulkan is off.
# Quit the Ollama tray app first. It binds 11434 and will ignore these settings.

$ErrorActionPreference = "Stop"
$Models = Join-Path $env:USERPROFILE ".ollama\models"
New-Item -ItemType Directory -Force -Path $Models | Out-Null

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

function Start-OllamaServer([string]$Name, [string]$Listen, [hashtable]$Extra) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "ollama"
    $psi.Arguments = "serve"
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.EnvironmentVariables["OLLAMA_HOST"] = $Listen
    $psi.EnvironmentVariables["OLLAMA_MODELS"] = $Models
    foreach ($key in $Extra.Keys) {
        $psi.EnvironmentVariables[$key] = $Extra[$key]
    }
    $process = [System.Diagnostics.Process]::Start($psi)
    Write-Host "Started $Name (pid $($process.Id)) on $Listen"
}

if (Test-PortOpen 11434) {
    Write-Host "Port 11434 is already in use. Quit Ollama from the system tray, then run this script again so the RX 6600 server can bind it."
} else {
    Start-OllamaServer "rx6600" "127.0.0.1:11434" @{
        CUDA_VISIBLE_DEVICES = "-1"
        OLLAMA_VULKAN = "1"
    }
}

if (Test-PortOpen 11435) {
    Write-Host "Port 11435 is already open."
} else {
    Start-OllamaServer "gtx1660" "127.0.0.1:11435" @{
        CUDA_VISIBLE_DEVICES = "0"
        OLLAMA_VULKAN = "0"
        GGML_VK_VISIBLE_DEVICES = "-1"
    }
}

Write-Host "Model folder: $Models"
Write-Host "Then run: valta models pull"
