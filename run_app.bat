@echo off
title RESOLVIX // Business Entity Resolution Platform
echo ============================================================
echo   Starting RESOLVIX Entity Resolution Full-Stack Platform
echo ============================================================
echo.
echo Initializing FastAPI backend server on http://localhost:8080
echo Open http://localhost:8080 in your browser to access the dashboard.
echo.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
pause
