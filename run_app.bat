@echo off
echo ============================================================
echo Starting Quantum Catalyst Platform
echo ============================================================
echo.

REM Activate conda environment
call conda activate quantum

REM Check if activation was successful
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment 'quantum'
    echo Please run: conda create -n quantum python=3.11 -y
    echo Then: conda install -c conda-forge pyscf qiskit qiskit-nature rdkit streamlit matplotlib py3dmol -y
    pause
    exit /b 1
)

echo [OK] Conda environment 'quantum' activated
echo.

REM Navigate to project directory
cd /d "a:\Quantum computing\Projects\Quantum_catalyst_platform"

REM Run Streamlit app
echo Starting Streamlit app...
echo.
streamlit run app.py

pause
