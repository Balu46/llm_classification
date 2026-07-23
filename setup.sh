#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Starting AutoQA PEFT Project Environment Setup ==="

# 1. Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3.10+ before running this script."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PYTHON_VERSION"

# 2. Setup Virtual Environment
if [ -d "venv" ]; then
    echo "Virtual environment 'venv' already exists. Skipping creation."
else
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
    echo "Virtual environment created successfully."
fi

# 3. Activate Virtual Environment
echo "Activating virtual environment..."
source venv/bin/activate

# 4. Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# 5. Install requirements
echo "Installing project dependencies from requirements.txt..."
pip install -r requirements.txt

# 6. Verify PyTorch CUDA availability
echo "Verifying PyTorch CUDA support..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

echo ""
echo "=== Environment Setup Completed Successfully! ==="
echo "To activate the environment in your terminal, run:"
echo "    source venv/bin/activate"
echo ""
echo "Note: Since Gemma 4 E2B is a gated model, remember to authorize on Hugging Face and login via:"
echo "    huggingface-cli login"
echo ""
