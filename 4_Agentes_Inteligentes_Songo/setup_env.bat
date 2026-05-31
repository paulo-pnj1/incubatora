@echo off
REM ============================================================
REM  SETUP DO AMBIENTE VIRTUAL - PROJECTO SONGO ML  (Windows)
REM  Universidade Kimpa Vita | Engenharia Informatica
REM ============================================================

echo ==============================================
echo   SETUP DO AMBIENTE VIRTUAL - SONGO ML
echo ==============================================

REM 1. Verificar Python
echo.
echo [1/4] Verificando Python...
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERRO: Python nao encontrado. Instale em https://python.org
    pause
    exit /b 1
)
python --version

REM 2. Criar ambiente virtual
echo.
echo [2/4] Criando ambiente virtual 'env_IA'...
python -m venv env_IA
IF NOT EXIST "env_IA\" (
    echo ERRO: Falha ao criar ambiente virtual.
    pause
    exit /b 1
)
echo   OK - Ambiente criado: env_IA\

REM 3. Activar e instalar dependencias
echo.
echo [3/4] Instalando dependencias...
call env_IA\Scripts\activate.bat

python -m pip install --upgrade pip --quiet
python -m pip install ^
    scikit-learn==1.4.2 ^
    pandas==2.2.2 ^
    numpy==1.26.4 ^
    matplotlib==3.8.4 ^
    seaborn==0.13.2 ^
    --quiet

echo   OK - Dependencias instaladas.

REM 4. Confirmar
echo.
echo [4/4] Verificacao final...
python -c "import sklearn, pandas, numpy, matplotlib, seaborn; print('    scikit-learn :', sklearn.__version__); print('    pandas       :', pandas.__version__); print('    numpy        :', numpy.__version__); print('    matplotlib   :', matplotlib.__version__); print('    seaborn      :', seaborn.__version__)"

echo.
echo ==============================================
echo   AMBIENTE PRONTO!
echo ==============================================
echo.
echo   Para activar o ambiente:
echo     env_IA\Scripts\activate
echo.
echo   Para treinar o modelo:
echo     python treinar_modelo_songo.py
echo.
echo   Para desactivar o ambiente:
echo     deactivate
echo ==============================================
pause
