FROM python:3.11-slim

WORKDIR /app

# Needed for python-magic on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*
    # System deps needed by Phase 5 worker:
# - ffmpeg/ffprobe for duration checks + segmentation + video->audio extraction
# - libgomp1/libstdc++6 are commonly needed by faster-whisper/ctranslate2 wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend code
COPY . /app

EXPOSE 8000

# If your FastAPI instance is in backend/app/main.py as `app`
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
