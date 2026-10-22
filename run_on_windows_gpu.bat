@echo off
REM One-click setup + full pipeline on a Windows laptop with an NVIDIA GPU.
REM Usage: double-click, or from a terminal in this folder:
REM     run_on_windows_gpu.bat "C:\Users\you\Downloads\Pork_Rasher_Error-Packaging-_v4i_yolov8.zip"
REM Needs Python 3.10-3.13 (64-bit) installed (can sit alongside newer versions) and a recent NVIDIA driver.
setlocal
cd /d "%~dp0"

set ZIP=%~1
if "%ZIP%"=="" if exist "%USERPROFILE%\Downloads\Pork Rasher Error-Packaging-.v4i.yolov8.zip" set "ZIP=%USERPROFILE%\Downloads\Pork Rasher Error-Packaging-.v4i.yolov8.zip"
if "%ZIP%"=="" if exist "%USERPROFILE%\Downloads\Pork_Rasher_Error-Packaging-_v4i_yolov8.zip" set "ZIP=%USERPROFILE%\Downloads\Pork_Rasher_Error-Packaging-_v4i_yolov8.zip"
echo Dataset zip: %ZIP%

REM PyTorch supports Python 3.10-3.13; pick one of those via the "py" launcher if installed.
set PYBASE=
for %%V in (3.12 3.13 3.11 3.10) do (
  if not defined PYBASE py -%%V -c "pass" >nul 2>&1 && set "PYBASE=py -%%V"
)
if not defined PYBASE (
  echo *** Python 3.10-3.13 not found. PyTorch has no builds for newer Python yet.
  echo *** Install Python 3.12 alongside your current one:   winget install Python.Python.3.12
  echo *** or download "Windows installer (64-bit)" for 3.12 from python.org, then run this again.
  goto :err
)
echo Using %PYBASE% for the project environment

REM Remove an environment created earlier with an unsupported Python.
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" || (
    echo Removing old .venv made with an unsupported Python ...
    rmdir /s /q .venv
  )
)
if not exist .venv (
  echo === Creating virtual environment ===
  %PYBASE% -m venv .venv || goto :err
)
call .venv\Scripts\activate.bat

python -c "import sys,struct; print('Python', sys.version.split()[0], struct.calcsize('P')*8, 'bit')"
python -c "import struct,sys; sys.exit(0 if struct.calcsize('P')==8 else 1)" || (
  echo *** This is 32-bit Python. PyTorch needs 64-bit Python: install the 64-bit version from python.org. ***
  goto :err
)

echo === Installing PyTorch with CUDA (about 2.5 GB, first run only) ===
python -m pip install --upgrade pip >nul
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul && goto :torch_ok
REM Newest CUDA build first (supports the newest Python versions); older ones as fallback.
for %%C in (cu128 cu126 cu124) do (
  echo --- trying PyTorch build %%C ---
  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/%%C && goto :torch_ok
)
echo *** No CUDA PyTorch build found for this Python version. Install Python 3.12 (64-bit) from python.org,
echo *** delete the .venv folder here, and run this script again. ***
goto :err
:torch_ok
echo === Installing project ===
pip install -e ".[dev]" || goto :err

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU found - will be VERY slow')"

echo === Running the full pipeline (about 30-60 min on an RTX 4070) ===
if "%ZIP%"=="" (
  python run_pipeline.py --device 0 || goto :err
) else (
  python run_pipeline.py --zip "%ZIP%" --device 0 || goto :err
)

echo.
echo Finished. Results are in the "results" folder; the summary is results\pork-rasher-reduced\SUMMARY.md
pause
exit /b 0

:err
echo.
echo *** Something failed - scroll up for the error message. ***
pause
exit /b 1
