FROM python:3.11-slim

WORKDIR /app

# System deps used by your parser (OCR + PDF + media)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libtesseract-dev \
    libleptonica-dev \
    libjpeg62-turbo \
    zlib1g \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optional but recommended
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ✅ COPY requirements FIRST
COPY requirements.txt /app/requirements.txt

# ✅ Then install
RUN pip install --no-cache-dir --default-timeout=180 --retries 10 -r /app/requirements.txt

# ✅ Then copy the rest
COPY . /app

EXPOSE 8020
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8020"]