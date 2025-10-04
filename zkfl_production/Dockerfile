FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/keys /app/circuits /app/proofs

# Set permissions
RUN chmod +x zkfl_production/run_server.py zkfl_production/run_client.py

# Expose port for Flower server
EXPOSE 8080

# Default command (can be overridden)
CMD ["python", "zkfl_production/run_server.py"]