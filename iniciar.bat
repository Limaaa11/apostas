@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  Modelo de Selecoes - Previsao Copa 2026
echo ============================================
echo.

REM Instala dependencias se necessario
if not exist ".venv\Scripts\waitress-serve.exe" (
    echo Instalando waitress...
    ".venv\Scripts\pip.exe" install waitress flask-limiter python-dotenv --quiet
)

echo Carregando modelo (primeira vez pode demorar 1-2 min)...
echo Servidor: Waitress (multi-thread, producao)
echo.

REM Abre o navegador apos 4 segundos (tempo para o modelo carregar)
start "" /b cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:5000"

".venv\Scripts\python.exe" app.py

pause
