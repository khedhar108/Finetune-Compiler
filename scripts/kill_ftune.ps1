Write-Host "Searching for ftune related processes..."

# Find processes with 'ftune' in the command line
$processes = Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*ftune*" -and $_.Name -eq "python.exe" }

if ($processes) {
    foreach ($proc in $processes) {
        Write-Host "Killing Process ID: $($proc.ProcessId) - $($proc.CommandLine)"
        Stop-Process -Id $proc.ProcessId -Force
    }
    Write-Host "All ftune instances have been terminated."
} else {
    Write-Host "No running ftune instances found."
}
