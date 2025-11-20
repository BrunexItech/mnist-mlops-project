#!/bin/bash
echo "Setting up MNIST MLOps environment..."
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
echo "Setup complete!"