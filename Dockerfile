FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Tesseract (y español opcional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /app/main.py
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

