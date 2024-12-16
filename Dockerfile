FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for models
RUN mkdir -p /app/savedModels

# Copy application files
COPY savedModels/ /app/savedModels/
COPY ml_service.py .

# Update model paths in the code
RUN sed -i 's|/savedModels/|/app/savedModels/|g' ml_service.py

EXPOSE 8080

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
