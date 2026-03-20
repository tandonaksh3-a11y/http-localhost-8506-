@echo off
echo ============================================
echo   AKRE TERMINAL - AI Quant Research Terminal
echo ============================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Launching AKRE TERMINAL...
streamlit run app.py --server.port=8501 --theme.base=dark
