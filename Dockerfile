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

# Create models directory and verify
RUN mkdir -p /savedModels && \
    echo "Created models directory"

# Copy models and verify
COPY savedModels/* /savedModels/
RUN ls -la /savedModels && \
    echo "Models directory contents:"

COPY ml_service.py .

EXPOSE 8080

CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "8080"]
