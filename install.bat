@echo off
echo Creating virtual environment...
python -m venv venv

echo Installing dependencies from requirements.txt...
call "%~dp0venv\Scripts\pip.exe" install -r "%~dp0requirements.txt"

echo.
echo Setup complete. You can now run the application using run_app.bat
pause
