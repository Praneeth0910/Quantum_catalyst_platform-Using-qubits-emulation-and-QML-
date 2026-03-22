@echo off
echo ============================================================
echo Starting Quantum Catalyst Platform
echo ============================================================
echo.

REM Navigate to project directory
cd /d "a:\Quantum computing\Projects\Quantum_catalyst_platform"

REM Activate local virtual environment
call .venv\Scripts\activate.bat

REM Run Streamlit app
echo Starting Streamlit app...
echo.
streamlit run app.py

pause