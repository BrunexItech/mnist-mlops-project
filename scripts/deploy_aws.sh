#!/bin/bash
echo "🚀 Deploying MNIST MLOps to AWS..."

# Build Docker images
echo "Building Docker images..."
docker build -t mnist-api -f infrastructure/docker/Dockerfile .
docker build -t mnist-streamlit -f infrastructure/docker/Dockerfile.streamlit .

echo "Build complete!"
echo "Next steps:"
echo "1. Set up AWS ECR repository"
echo "2. Push images to ECR"
echo "3. Deploy to ECS or EKS"