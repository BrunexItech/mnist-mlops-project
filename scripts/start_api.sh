#!/bin/bash
echo "Starting FastAPI server..."
uvicorn src.serving.api.main:app --reload --host 0.0.0.0 --port 8000