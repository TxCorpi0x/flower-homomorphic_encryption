#!/bin/bash
# Quick test deployment with lightweight configuration

set -e

echo "🧪 ZKFL Production Test Deployment"
echo "===================================="

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f zkfl_production/docker-compose.yml down 2>/dev/null || true

# Build latest images
echo "🏗️  Building images..."
docker build -f zkfl_production/Dockerfile.server -t zkfl-server-test .
docker build -f zkfl_production/Dockerfile.client -t zkfl-client-test .

# Test server alone first
echo "🚀 Testing server startup..."
docker run --rm --name zkfl-test-server -d \
  -p 8080:8080 \
  --memory="2g" \
  -e DATASET_TYPE=synthetic \
  -e NUM_CLIENTS=2 \
  zkfl-server-test

echo "⏳ Waiting for server to initialize..."
sleep 20

echo "📋 Server logs:"
docker logs zkfl-test-server

# Check if server is still running
if docker ps --filter name=zkfl-test-server --format '{{.Names}}' | grep -q zkfl-test-server; then
    echo "✅ Server is running successfully!"
    
    echo "🧪 Testing client connection..."
    docker run --rm --name zkfl-test-client \
      --link zkfl-test-server:server \
      --memory="1g" \
      -e DATASET_TYPE=synthetic \
      -e SERVER_ADDRESS=server:8080 \
      zkfl-client-test &
    
    CLIENT_PID=$!
    sleep 10
    
    echo "📋 Client logs:"
    docker logs zkfl-test-client 2>/dev/null || echo "Client container not found"
    
    # Clean up
    docker stop zkfl-test-client 2>/dev/null || true
    docker stop zkfl-test-server
    
    echo "✅ Test deployment completed successfully!"
else
    echo "❌ Server failed to start properly"
    docker logs zkfl-test-server
    docker stop zkfl-test-server 2>/dev/null || true
    exit 1
fi