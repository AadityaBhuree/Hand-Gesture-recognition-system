# ==============================================================================
# MNIST Neural Digit Laboratory - Production Microservice Dockerfile
# Hardened Python 3.10-slim container with WSGI Gunicorn server
# ==============================================================================

FROM python:3.10-slim

WORKDIR /app

# System dependencies for image processing and network health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create unprivileged service user
RUN groupadd -g 10002 digitgroup && \
    useradd -u 10002 -g digitgroup -s /bin/bash -m digituser && \
    mkdir -p /app/models && \
    chown -R digituser:digitgroup /app

# Copy application artifacts
COPY --chown=digituser:digitgroup app.py /app/app.py
COPY --chown=digituser:digitgroup services /app/services
COPY --chown=digituser:digitgroup templates /app/templates
COPY --chown=digituser:digitgroup models /app/models

USER digituser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app:app"]
