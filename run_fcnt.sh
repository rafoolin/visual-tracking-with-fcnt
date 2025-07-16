#!/bin/bash

echo "🔧 Setting up FCNT tracking environment with Conda..."

# 1. Create conda environment
ENV_NAME="fcnt_env"
PYTHON_VERSION="3.10"

echo "📦 Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# 2. Activate the environment
echo "✅ Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 3. Install Python dependencies
echo "📚 Installing required packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the code
cd fcn_tracker

python run.py --config ./configs/config.yaml