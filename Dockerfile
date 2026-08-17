# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend + Serving
FROM python:3.11-slim

WORKDIR /app


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, models, and metadata
COPY src/ ./src/
COPY main.py .
COPY checkpoints/ ./checkpoints/
COPY indices/ ./indices/
COPY metrics/ ./metrics/
RUN mkdir -p data
RUN python -c "from src.dataset import download_and_extract_cub200; download_and_extract_cub200('data')"



# Copy compiled React frontend from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# HuggingFace Spaces exposes port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
