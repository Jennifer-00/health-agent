@echo off
chcp 65001 >nul
title Health Agent Launcher
cd /d %~dp0

echo [1/4] Starting infrastructure (Redis, Neo4j, Qdrant)...
docker compose up -d redis neo4j qdrant
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Docker Compose failed. Make sure Docker Desktop is running.
    pause
    exit /b 1
)

echo [2/4] Waiting for services to initialize...
timeout /t 10 /nobreak >nul

echo [3/4] Starting backend (port 8000)...
start "Backend - Health Agent" cmd /k "cd /d %~dp0 && uvicorn backend.main:app --reload --port 8000"

echo [4/4] Starting frontend (port 3000)...
start "Frontend - Health Agent" cmd /k "cd /d %~dp0\frontend && npm install --prefer-offline && npm run dev"

echo.
echo ==========================================
echo  Health Agent started!
echo  Backend:  http://localhost:8000/health
echo  Frontend: http://localhost:3000
echo ==========================================
echo.
echo Press any key to shut down all services...
pause >nul

echo Stopping Docker services...
docker compose down
