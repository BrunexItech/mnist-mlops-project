from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ...utils.config import config

app = FastAPI(title="MNIST Classification API", version="1.0.0")

# Load model
model = None

class PredictionRequest(BaseModel):
    image: list

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: list

def load_model():
    """Load the trained model"""
    global model
    model_path = config.model_paths['deployed'] / 'mnist_cnn_model.keras'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {"message": "MNIST Classification API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert to numpy array and preprocess
        image_array = np.array(request.image).reshape(1, 28, 28, 1)
        image_array = image_array.astype('float32') / 255.0
        
        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        probabilities = predictions[0].tolist()
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)