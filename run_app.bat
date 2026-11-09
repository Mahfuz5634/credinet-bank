@echo off
title CrediNet - Banking Loan Officer Portal
echo ======================================================================
echo   CrediNet: Automated Loan Approval and Credit Risk Assessment
echo   Student: Md. Mahfuz ^| ID: 0432310005101057 ^| CSE 422
echo ======================================================================
echo.

set PYTHON_CMD=python
if exist "venv\Scripts\python.exe" (
    set PYTHON_CMD="venv\Scripts\python.exe"
)

echo [INFO] Checking Python environment...
%PYTHON_CMD% -c "import flask, torch, sklearn" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Required packages (PyTorch, Flask, etc.) not detected!
    echo [INFO] Installing required packages from requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Installation failed. Please ensure Python and internet are available.
        pause
        exit /b 1
    )
)

if not exist "models\credinet_deep_mlp.pth" (
    echo [INFO] Pre-trained model not found. Training PyTorch Deep MLP...
    %PYTHON_CMD% train_models.py
)

echo.
echo [INFO] Starting CrediNet Banking Web Portal...
echo [INFO] Opening http://127.0.0.1:5000 in your browser...
echo.
start http://127.0.0.1:5000
%PYTHON_CMD% app.py
pause
