FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Önce sadece requirements kopyala (layer cache için)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyaları
COPY src/ ./src/
COPY outputs/checkpoints/best_model.pt ./outputs/checkpoints/best_model.pt

WORKDIR /app/src

ENV CORS_ORIGINS="*"

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
