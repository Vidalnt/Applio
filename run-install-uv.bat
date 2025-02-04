@echo off
setlocal enabledelayedexpansion
title Applio Installer

echo Welcome to the Applio Installer!
echo.

set "INSTALL_DIR=%cd%"
set "MINICONDA_DIR=%UserProfile%\Miniconda3"
set "ENV_DIR=%INSTALL_DIR%\env"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Windows-x86_64.exe"
set "CONDA_EXE=%MINICONDA_DIR%\Scripts\conda.exe"

call :cleanup
call :install_miniconda
call :create_conda_env
call :install_dependencies

echo Applio has been installed successfully!
echo To start Applio, please run 'run-applio.bat'.
echo.
pause
exit /b 0

:cleanup
echo Cleaning up unnecessary files...
for %%F in (Makefile Dockerfile docker-compose.yaml *.sh) do if exist "%%F" del "%%F"
echo Cleanup complete.
echo.
exit /b 0

:install_miniconda
if exist "%CONDA_EXE%" (
    echo Miniconda already installed. Skipping installation.
    exit /b 0
)

echo Miniconda not found. Starting download and installation...
powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile 'miniconda.exe'}"
if not exist "miniconda.exe" goto :download_error

start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_DIR%
if errorlevel 1 goto :install_error

del miniconda.exe
echo Miniconda installation complete.
echo.
exit /b 0

:create_conda_env
echo Creating Conda environment...
call "%MINICONDA_DIR%\_conda.exe" create --no-shortcuts -y -k --prefix "%ENV_DIR%" python=3.10
if errorlevel 1 goto :error
echo Conda environment created successfully.
echo.

if exist "%ENV_DIR%\python.exe" (
    echo Installing uv package installer...
    "%ENV_DIR%\python.exe" -m pip install uv
    if errorlevel 1 goto :error
    echo uv installation complete.
    echo.
)
exit /b 0

:install_dependencies
echo Installing dependencies...
call "%MINICONDA_DIR%\condabin\conda.bat" activate "%ENV_DIR%" || goto :error
uv pip install --upgrade setuptools || goto :error
uv pip install --link-mode=copy -r "%INSTALL_DIR%\requirements.txt" || goto :error
uv pip install --link-mode=copy torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --upgrade --index-url https://download.pytorch.org/whl/cu121 || goto :error
uv pip install --link-mode=copy numpy==1.23.5 || goto : error
call "%MINICONDA_DIR%\condabin\conda.bat" deactivate
echo Dependencies installation complete.
echo.
exit /b 0

:download_error
echo Download failed. Please check your internet connection and try again.
goto :error

:install_error
echo Miniconda installation failed.
goto :error

:error
echo An error occurred during installation. Please check the output above for details.
pause
exit /b 1
