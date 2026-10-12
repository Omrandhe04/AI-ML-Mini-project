@echo off
title RadVision AI - Chest X-Ray CAD Server
echo ====================================================================
echo   RadVision AI - Chest X-Ray CAD & Heatmap Localization Studio
echo   Launching Localhost Server connected to SQLite Database...
echo ====================================================================
echo.

if exist "C:\msys64\ucrt64\bin\python3.exe" (
    "C:\msys64\ucrt64\bin\python3.exe" app.py
) else (
    python app.py
)

pause
