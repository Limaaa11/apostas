@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  Modelo de Selecoes - Previsao e Banca
echo ============================================
echo.
echo Iniciando servidor... aguarde a mensagem "Pronto!"
echo Depois o navegador abre sozinho em http://localhost:5000
echo (Para encerrar: feche esta janela)
echo.
start "" http://localhost:5000
".venv\Scripts\python.exe" app.py
pause
