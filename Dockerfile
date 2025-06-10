FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    espeak-ng \
    build-essential \
    pkg-config \
    libicu-dev \
    ffmpeg \
    gettext \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and model download script first to leverage Docker cache
COPY requirements.txt .
COPY download_model.py .

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support (for NVIDIA GPU)
# This ensures GPU-enabled versions are prioritized.
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies from requirements.txt
# (torch and torchaudio are already installed, pip will skip them if listed, or handle if not)
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Run the model and NLTK data download script
# Pipe "y" to auto-accept Coqui TTS license prompt if it appears
RUN echo "y" | python download_model.py

# Copy the rest of the application code
COPY . .

# Create necessary directories that app.py will use
# Ensure they are writable if Gunicorn runs as non-root (default is root)
RUN mkdir -p /app/static/chunk_audio /app/uploads_temp/speaker_samples && \
    chmod -R 777 /app/static /app/uploads_temp
    # Using 777 for simplicity in dev; use more restrictive permissions for production.

EXPOSE 5000

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "600", "app:app"]