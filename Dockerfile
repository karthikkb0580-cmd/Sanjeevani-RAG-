# =============================================================================
# Sanjeevani AI – RAG Service Dockerfile
# Multi-stage build for lean production image
# =============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies into a prefix for easy copying
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="Sanjeevani AI Team"
LABEL description="Sanjeevani RAG Service"
LABEL version="1.0.0"

# Create non-root user
RUN groupadd -r rag && useradd -r -g rag -d /app -s /sbin/nologin raguser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Create upload directory with correct permissions
RUN mkdir -p /tmp/rag_uploads && chown -R raguser:rag /tmp/rag_uploads

# Switch to non-root user
USER raguser

# Environment defaults (overridden by docker-compose or Kubernetes secrets)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
