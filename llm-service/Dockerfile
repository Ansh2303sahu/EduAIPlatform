FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install deps first for better cache usage
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY . /app

EXPOSE 8030

# Default (compose will override with --reload in dev)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8030"]