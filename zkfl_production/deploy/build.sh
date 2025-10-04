#!/bin/bash

# Build script for ZKFL production environment
# This script builds Docker images and prepares the environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="zkfl"
VERSION=${1:-"latest"}
REGISTRY=${REGISTRY:-"localhost:5000"}
BUILD_ARGS=""

echo -e "${GREEN}🔨 Building ZKFL Production Environment${NC}"
echo "Version: $VERSION"
echo "Registry: $REGISTRY"
echo ""

# Function to print status
print_status() {
    echo -e "${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if Docker is running
if ! /usr/local/bin/docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p zkfl_production/{logs,data,keys,circuits,proofs}
print_success "Directories created"

# Build base Dockerfile (if exists)
if [ -f "zkfl_production/Dockerfile" ]; then
    print_status "Building base ZKFL image..."
    /usr/local/bin/docker build -f zkfl_production/Dockerfile -t ${PROJECT_NAME}/base:${VERSION} .
    print_success "Base image built: ${PROJECT_NAME}/base:${VERSION}"
fi

# Build server image
print_status "Building ZKFL server image..."
/usr/local/bin/docker build -f zkfl_production/Dockerfile.server -t ${PROJECT_NAME}/server:${VERSION} .
if [ $? -eq 0 ]; then
    print_success "Server image built: ${PROJECT_NAME}/server:${VERSION}"
else
    print_error "Failed to build server image"
    exit 1
fi

# Build client image
print_status "Building ZKFL client image..."
/usr/local/bin/docker build -f zkfl_production/Dockerfile.client -t ${PROJECT_NAME}/client:${VERSION} .
if [ $? -eq 0 ]; then
    print_success "Client image built: ${PROJECT_NAME}/client:${VERSION}"
else
    print_error "Failed to build client image"
    exit 1
fi

# Tag images for registry if specified
if [ "$REGISTRY" != "localhost:5000" ] && [ "$REGISTRY" != "" ]; then
    print_status "Tagging images for registry: $REGISTRY"
    
    /usr/local/bin/docker tag ${PROJECT_NAME}/server:${VERSION} ${REGISTRY}/${PROJECT_NAME}/server:${VERSION}
    /usr/local/bin/docker tag ${PROJECT_NAME}/client:${VERSION} ${REGISTRY}/${PROJECT_NAME}/client:${VERSION}
    
    print_success "Images tagged for registry"
    
    # Push to registry if requested
    if [ "$PUSH" = "true" ]; then
        print_status "Pushing images to registry..."
        /usr/local/bin/docker push ${REGISTRY}/${PROJECT_NAME}/server:${VERSION}
        /usr/local/bin/docker push ${REGISTRY}/${PROJECT_NAME}/client:${VERSION}
        print_success "Images pushed to registry"
    fi
fi

# Display built images
print_status "Built images:"
/usr/local/bin/docker images | grep ${PROJECT_NAME} | grep ${VERSION}

echo ""
print_success "🎉 Build completed successfully!"
echo ""
echo "Next steps:"
echo "  • Development: docker-compose -f zkfl_production/docker-compose.dev.yml up"
echo "  • Production:  docker-compose -f zkfl_production/docker-compose.yml up"
echo "  • Kubernetes:  kubectl apply -f zkfl_production/k8s/"
echo ""

# Generate environment file template
if [ ! -f "zkfl_production/.env" ]; then
    print_status "Generating environment template..."
    cat > zkfl_production/.env << EOF
# ZKFL Production Environment Configuration

# Server Configuration
NUM_ROUNDS=10
MIN_NUM_CLIENTS=3
MIN_AVAILABLE_CLIENTS=3
MIN_FIT_CLIENTS=3
MIN_EVAL_CLIENTS=2

# Security
ZK_ENABLED=true
DP_ENABLED=true
BLOCKCHAIN_ENABLED=false

# Privacy Settings
DP_EPSILON=1.0
DP_DELTA=1e-5
DP_CLIPPING_NORM=1.0

# Performance
BATCH_SIZE=32
LEARNING_RATE=0.01

# Registry (for Kubernetes deployments)
REGISTRY=${REGISTRY}
VERSION=${VERSION}
EOF
    print_success "Environment template created: zkfl_production/.env"
fi