FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y el modelo
COPY . .

# Crear directorio para el modelo si no existe
RUN mkdir -p model

# Variables de entorno por defecto
ENV MODEL_PATH=model/bird_classifier.onnx
ENV LABELS_PATH=model/bird_labels.txt
ENV PYTHONUNBUFFERED=1

# Render asigna automáticamente el puerto
ENV PORT=10000

# Exponer el puerto
EXPOSE 10000

# Comando para ejecutar la aplicación
CMD gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120