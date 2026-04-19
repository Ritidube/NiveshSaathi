# ──────────────────────────────────────────────────────────────────────────────
# NiveshSaathi — Dockerfile for HuggingFace Spaces
# Platform: HuggingFace Spaces (Docker SDK)
# Port: 7860 (required by HF Spaces)
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# HuggingFace Spaces runs as user 1000 — set this up
RUN useradd -m -u 1000 user
WORKDIR /app

# System libraries needed by OpenCV + EasyOCR
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (faster rebuilds via Docker layer cache)
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY --chown=user . .

# Create directories that must exist at runtime
RUN mkdir -p static/gan_cache rag_db && \
    chown -R user:user /app

# Switch to non-root user (required by HuggingFace)
USER user

# Expose port 7860 (HuggingFace Spaces requirement)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the app
CMD ["python", "app.py"]