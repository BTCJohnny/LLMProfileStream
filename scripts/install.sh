#!/bin/bash

echo "Setting up LLMProfileStream project directories..."

mkdir -p llm_helpers
mkdir -p data/summaries
mkdir -p db
mkdir -p config
mkdir -p logs
mkdir -p tests
mkdir -p scripts

echo "Creating Python virtual environment (if not exists)..."
python3 -m venv venv

echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
echo "To activate your environment, run: source venv/bin/activate"
echo "To run the pipeline: python ohlcv_profile_continious.py"