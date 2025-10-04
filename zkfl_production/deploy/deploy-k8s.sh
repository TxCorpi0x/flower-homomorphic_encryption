#!/bin/bash

# Kubernetes deployment script for ZKFL production environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="zkfl-system"
ACTION=${1:-"deploy"}
REGISTRY=${REGISTRY:-"localhost:5000"}
VERSION=${VERSION:-"latest"}

print_status() {
    echo -e "${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

echo -e "${GREEN}☸️  ZKFL Kubernetes Deployment${NC}"
echo "Namespace: $NAMESPACE"
echo "Action: $ACTION"
echo "Registry: $REGISTRY"
echo "Version: $VERSION"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if we can connect to cluster
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

case $ACTION in
    "deploy"|"apply")
        print_status "Deploying ZKFL to Kubernetes..."
        
        # Update image tags in manifests if using custom registry/version
        if [ "$REGISTRY" != "localhost:5000" ] || [ "$VERSION" != "latest" ]; then
            print_status "Updating image references..."
            
            # Create temporary manifests with updated image references
            mkdir -p /tmp/zkfl-k8s
            cp -r zkfl_production/k8s/* /tmp/zkfl-k8s/
            
            # Update server deployment
            sed -i.bak "s|image: zkfl/server:latest|image: ${REGISTRY}/zkfl/server:${VERSION}|g" /tmp/zkfl-k8s/server-deployment.yaml
            
            # Update client deployment  
            sed -i.bak "s|image: zkfl/client:latest|image: ${REGISTRY}/zkfl/client:${VERSION}|g" /tmp/zkfl-k8s/client-deployment.yaml
            
            MANIFEST_DIR="/tmp/zkfl-k8s"
        else
            MANIFEST_DIR="zkfl_production/k8s"
        fi
        
        # Apply manifests in order
        kubectl apply -f ${MANIFEST_DIR}/namespace.yaml
        kubectl apply -f ${MANIFEST_DIR}/configmap.yaml
        kubectl apply -f ${MANIFEST_DIR}/server-deployment.yaml
        kubectl apply -f ${MANIFEST_DIR}/client-deployment.yaml
        
        print_success "Manifests applied"
        
        # Wait for server to be ready
        print_status "Waiting for server to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment/zkfl-server -n $NAMESPACE
        
        # Wait for clients to be ready
        print_status "Waiting for clients to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment/zkfl-clients -n $NAMESPACE
        
        print_success "Deployment completed!"
        
        # Show status
        echo ""
        print_info "Deployment status:"
        kubectl get all -n $NAMESPACE
        
        # Get service endpoint
        echo ""
        print_info "Service endpoints:"
        kubectl get svc -n $NAMESPACE
        
        # Show how to access the service
        echo ""
        print_info "Access instructions:"
        echo "  • Port forward: kubectl port-forward svc/zkfl-server-service 8080:8080 -n $NAMESPACE"
        echo "  • Logs (server): kubectl logs -f deployment/zkfl-server -n $NAMESPACE"
        echo "  • Logs (clients): kubectl logs -f deployment/zkfl-clients -n $NAMESPACE"
        
        # Clean up temporary files
        if [ -d "/tmp/zkfl-k8s" ]; then
            rm -rf /tmp/zkfl-k8s
        fi
        ;;
        
    "delete"|"remove")
        print_status "Removing ZKFL from Kubernetes..."
        kubectl delete namespace $NAMESPACE --ignore-not-found=true
        print_success "ZKFL removed from cluster"
        ;;
        
    "status")
        print_status "ZKFL status in Kubernetes:"
        if kubectl get namespace $NAMESPACE &> /dev/null; then
            kubectl get all -n $NAMESPACE
            echo ""
            print_info "Recent events:"
            kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | tail -10
        else
            print_info "ZKFL is not deployed"
        fi
        ;;
        
    "logs")
        SERVICE=${2:-"server"}
        case $SERVICE in
            "server")
                print_status "Server logs:"
                kubectl logs -f deployment/zkfl-server -n $NAMESPACE
                ;;
            "client"|"clients")
                print_status "Client logs:"
                kubectl logs -f deployment/zkfl-clients -n $NAMESPACE
                ;;
            "all")
                print_status "All logs:"
                kubectl logs -f deployment/zkfl-server -n $NAMESPACE &
                kubectl logs -f deployment/zkfl-clients -n $NAMESPACE &
                wait
                ;;
            *)
                print_error "Unknown service: $SERVICE"
                echo "Available services: server, client, all"
                exit 1
                ;;
        esac
        ;;
        
    "scale")
        REPLICAS=${2:-3}
        print_status "Scaling clients to $REPLICAS replicas..."
        kubectl scale deployment zkfl-clients --replicas=$REPLICAS -n $NAMESPACE
        print_success "Clients scaled to $REPLICAS replicas"
        ;;
        
    "port-forward")
        PORT=${2:-8080}
        print_status "Port forwarding server to localhost:$PORT..."
        kubectl port-forward svc/zkfl-server-service $PORT:8080 -n $NAMESPACE
        ;;
        
    "config")
        print_status "Current configuration:"
        kubectl get configmap zkfl-config -n $NAMESPACE -o yaml
        ;;
        
    *)
        print_error "Unknown action: $ACTION"
        echo ""
        echo "Usage: $0 [action] [options]"
        echo ""
        echo "Actions:"
        echo "  deploy/apply     - Deploy ZKFL to cluster"
        echo "  delete/remove    - Remove ZKFL from cluster"
        echo "  status           - Show deployment status"
        echo "  logs [service]   - Show logs (server|client|all)"
        echo "  scale [replicas] - Scale client replicas"
        echo "  port-forward [port] - Port forward server"
        echo "  config           - Show current configuration"
        echo ""
        echo "Environment variables:"
        echo "  REGISTRY - Container registry (default: localhost:5000)"
        echo "  VERSION  - Image version (default: latest)"
        exit 1
        ;;
esac