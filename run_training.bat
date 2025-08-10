@echo off

echo ========================================
echo  Thai SLM MoE Training Pipeline
echo ========================================

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies
    pause
    exit /b 1
)

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%cd%

REM Check system capabilities
echo.
echo Checking system capabilities...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {\"cuda\" if torch.cuda.is_available() else \"cpu\"}')"

REM Download and prepare dataset
echo.
echo Preparing Thai dataset...
python dataset.py
if errorlevel 1 (
    echo Failed to prepare dataset
    pause
    exit /b 1
)

REM Train the model
echo.
echo Starting training...
echo Note: Training on CPU will be slower but functional
echo Training will automatically adjust for your hardware
python train.py
if errorlevel 1 (
    echo Training failed
    pause
    exit /b 1
)

REM Test the model
echo.
echo Testing the model...
python inference.py
if errorlevel 1 (
    echo Model testing failed (model may not be fully trained)
    echo This is normal for CPU training with limited epochs
)

echo.
echo ========================================
echo  Training Pipeline Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run 'python gradio_app.py' for web interface
echo 2. Run 'python evaluate.py' for model evaluation
echo 3. Set HF_TOKEN and run 'python upload_to_hf.py' to upload to Hugging Face
echo.
pause
