FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive

# + libgl1 para evitar "libGL.so.1 not found"
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
EXPOSE 8000
# usa forma shell para expandir ${PORT}
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}

