# PowerShell Launcher for SPIRE Streamlit App
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "          SPIRE: NeMo Conformer Kannada Dialect AI Launcher           " -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Activating Virtual Environment (.venv)..." -ForegroundColor Green

& ".\.venv\Scripts\Activate.ps1"

Write-Host "Launching Streamlit Presentation UI..." -ForegroundColor Cyan
& ".\.venv\Scripts\streamlit.exe" run Frontend\app.py --server.port 8501
