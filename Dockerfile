FROM python:3.11-slim

# Minimal OS deps (awscli optional; keep if you need it at runtime)
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends awscli curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 10001 appuser

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py
COPY README.md /app/README.md
COPY src/ /app/src/
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime files
COPY app.py /app/app.py
COPY templates/ /app/templates/
COPY model/ /app/model/

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python3", "app.py"]
