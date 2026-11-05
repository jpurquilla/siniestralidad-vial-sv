
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias del sistema para XGBoost y WeasyPrint
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libharfbuzz0b \
        libfontconfig1 \
        libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Instalar las dependencias
COPY api/requirements.txt /app/api/requirements.txt

RUN pip install --no-cache-dir \
    -r /app/api/requirements.txt

# Código y artefactos necesarios
COPY api/app /app/api/app
COPY models /app/models
COPY data /app/data
COPY reports /app/reports

# Ejecutar desde api para resolver los imports
WORKDIR /app/api

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
