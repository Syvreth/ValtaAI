# Stop the Ollama servers Valta started on ports 11434 and 11435, so start-ollama.ps1 can
# start them again in another mode. Only processes named ollama that listen on those ports
# are stopped. Models stay on disk.

$ports = 11434, 11435
$stopped = 0
$failed = 0
foreach ($port in $ports) {
    $owners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($owner in $owners) {
        $process = Get-Process -Id $owner -ErrorAction SilentlyContinue
        if ($null -eq $process) { continue }
        if ($process.ProcessName -notlike "ollama*") {
            Write-Host "Port $port belongs to $($process.ProcessName) (pid $owner), not Ollama. Left running." -ForegroundColor Yellow
            continue
        }
        try {
            Stop-Process -Id $owner -Confirm:$false -ErrorAction Stop
            Write-Host "Stopped Ollama on port $port (pid $owner)."
            $stopped++
        } catch {
            $failed++
            Write-Host "Could not stop Ollama on port $port (pid $owner): $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}
if ($failed -gt 0) {
    Write-Host "Access denied usually means Ollama was started from an administrator window."
    Write-Host "Run this script from an administrator PowerShell, or end ollama.exe in Task Manager."
} elseif ($stopped -eq 0) {
    Write-Host "No Ollama server was listening on 11434 or 11435."
}
