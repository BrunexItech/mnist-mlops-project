FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY config/ ./config/
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "src.serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]