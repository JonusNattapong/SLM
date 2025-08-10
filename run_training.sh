#!/bin/bash

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Download and prepare dataset
echo "Preparing Thai dataset..."
python dataset.py

# Train the model
echo "Starting training..."
python train.py

# Test the model
echo "Testing the model..."
python inference.py

# Upload to Hugging Face (optional)
echo "To upload to Hugging Face:"
echo "1. Set your HF_TOKEN environment variable"
echo "2. Run: python upload_to_hf.py"

echo "Training completed!"
