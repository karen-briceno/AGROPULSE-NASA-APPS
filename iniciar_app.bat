@echo off
chcp 65001 >nul
title AgroPulse — Iniciando...

cd /d "%~dp0"

echo.
echo  =========================================================
echo   AgroPulse  ^|  Story County, Ames, Iowa
echo   Grilla 30m  ^|  360 Pixeles  ^|  NASA Space Apps
echo  =========================================================
echo.
echo  Iniciando servidor Streamlit...
echo  Abre tu navegador en: http://localhost:8501
echo.
echo  [Presiona Ctrl+C para detener la aplicacion]
echo.

"C:\Users\USUARIO\AppData\Local\Python\pythoncore-3.14-64\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless false

pause
