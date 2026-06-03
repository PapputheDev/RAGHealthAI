# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (kept minimal). Some ML/NLP deps may require build tools; add only if needed.
RUN apt-get update \
    ; apt-get install -y --no-install-recommends ca-certificates \
    ; rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    ; pip install -r /app/requirements.txt

# Copy application code, UI, and synthetic KB docs.
COPY app/    /app/app/
COPY data/   /app/data/
COPY static/ /app/static/

# Create runtime directories and a non-root user.
RUN mkdir -p /app/chroma_db \
    ; adduser --disabled-password --gecos "" appuser \
    ; chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Single worker — conversation memory is in-process; scale via external session store for prod
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
