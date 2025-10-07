@echo off
rem Set Python to use UTF-8 encoding for all file I/O
set PYTHONUTF8=1
echo Starting YOLO Auto Trainer...
"%~dp0venv\Scripts\python.exe" "%~dp0run_training_ui.py"
