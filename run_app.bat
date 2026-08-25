@echo off
TITLE SPIRE - Kannada Dialect Classifier Streamlit App
echo ======================================================================
echo           SPIRE: NeMo Conformer Kannada Dialect AI Launcher           
echo ======================================================================
echo.
echo Activating Virtual Environment (.venv)...
call .venv\Scripts\activate.bat
echo.
echo Launching Streamlit App...
echo URL will open automatically in your browser...
echo.
streamlit run Frontend\app.py --server.port 8501 --server.headless false
pause
