# Launcher for background model download
Set-Location -Path "D:\Project\DataMate"
"Starting launch_download.ps1 at $(Get-Date)" | Out-File -FilePath "D:\Project\DataMate\launcher_debug.log"
try {
    & "D:\Project\DataMate\.venv\Scripts\python.exe" -u "D:\Project\DataMate\download_model.py" 2>&1 | Tee-Object -FilePath "D:\Project\DataMate\download_model.log"
} catch {
    $_ | Out-File -FilePath "D:\Project\DataMate\launcher_debug.log" -Append
}
