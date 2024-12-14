# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install git and other dependencies
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-ml.txt

# Clone the repository and copy models
RUN git clone https://github.com/your-username/Dis-Easify_Deploy_Source.git repo && \
    mkdir -p /app/models && \
    cp -r repo/savedModels/* /app/models/ && \
    rm -rf repo

# Copy the FastAPI application code
COPY ml_service.py .

# Expose port
EXPOSE 8080

# Start the FastAPI server
CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "8080"]
