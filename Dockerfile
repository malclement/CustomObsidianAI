# Use Python 3.11 slim as the base image for the build stage
FROM python:3.11-slim AS builder

# Set work directory for the build stage
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Don't use a token for public models by default
    TRANSFORMERS_OFFLINE=0 \
    USE_AUTH=0

# Install system dependencies required for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Use the same slim base image for the final stage
FROM python:3.11-slim

# Set work directory for the final stage
WORKDIR /app

# Set environment variables for runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    ENVIRONMENT=production \
    DEBUG=false \
    # Hugging Face settings
    TRANSFORMERS_OFFLINE=0 \
    USE_AUTH=0 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1

# Create non-root user for security
RUN adduser --disabled-password --gecos "" appuser

# Install only the runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libblas-dev \
    liblapack-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create necessary directories
RUN mkdir -p /app/logs /app/prompts /app/models && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser ./app /app/app
COPY --chown=appuser:appuser ./prompts /app/prompts

# Add fix script for handling authorization issues
COPY --chown=appuser:appuser ./fix_auth.sh /app/fix_auth.sh
RUN chmod +x /app/fix_auth.sh

# Switch to non-root user
USER appuser

# Set Python path
ENV PYTHONPATH=/app

# Set cache directory for huggingface
ENV HF_HOME=/app/models/huggingface
RUN mkdir -p $HF_HOME && chmod 755 $HF_HOME

# Create volume for persistent data
VOLUME ["/app/logs", "/app/models", "/app/prompts"]

# Expose application port
EXPOSE 8000

# Set startup script that will handle auth settings
COPY --chown=appuser:appuser ./startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Set default command
ENTRYPOINT ["/app/startup.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
