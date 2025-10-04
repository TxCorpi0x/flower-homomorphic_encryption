#!/bin/bash

# Docker deployment script for ZKFL production environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE=${1:-"zkfl_production/docker-compose.yml"}
PROJECT_NAME="zkfl"
ACTION=${2:-"up"}

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

echo -e "${GREEN}🚀 ZKFL Docker Deployment${NC}"
echo "Compose file: $COMPOSE_FILE"
echo "Action: $ACTION"
echo ""

# Check if Docker and Docker Compose are available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed or not in PATH"
    exit 1
fi

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    print_error "Compose file not found: $COMPOSE_FILE"
    exit 1
fi

# Load environment variables if .env file exists
if [ -f "zkfl_production/.env" ]; then
    print_status "Loading environment variables..."
    export $(cat zkfl_production/.env | grep -v '^#' | xargs)
    print_success "Environment variables loaded"
fi

# Create necessary directories
print_status "Creating required directories..."
mkdir -p zkfl_production/{logs,data,keys,circuits,proofs}
print_success "Directories created"

case $ACTION in
    "up"|"start")
        print_status "Starting ZKFL services..."
        docker-compose -f $COMPOSE_FILE up -d
        
        if [ $? -eq 0 ]; then
            print_success "Services started successfully"
            
            # Wait for server to be ready
            print_status "Waiting for server to be ready..."
            for i in {1..30}; do
                if curl -s http://localhost:8080/health > /dev/null 2>&1; then
                    print_success "Server is ready!"
                    break
                fi
                echo -n "."
                sleep 2
            done
            
            echo ""
            print_info "Services status:"
            docker-compose -f $COMPOSE_FILE ps
            
            echo ""
            print_info "Access points:"
            echo "  • Server API: http://localhost:8080"
            echo "  • Server Health: http://localhost:8080/health"
            echo "  • Logs: docker-compose -f $COMPOSE_FILE logs -f"
            
        else
            print_error "Failed to start services"
            exit 1
        fi
        ;;
        
    "down"|"stop")
        print_status "Stopping ZKFL services..."
        docker-compose -f $COMPOSE_FILE down
        print_success "Services stopped"
        ;;
        
    "restart")
        print_status "Restarting ZKFL services..."
        docker-compose -f $COMPOSE_FILE down
        docker-compose -f $COMPOSE_FILE up -d
        print_success "Services restarted"
        ;;
        
    "logs")
        print_status "Showing logs..."
        docker-compose -f $COMPOSE_FILE logs -f
        ;;
        
    "ps"|"status")
        print_status "Service status:"
        docker-compose -f $COMPOSE_FILE ps
        ;;
        
    "build")
        print_status "Building services..."
        docker-compose -f $COMPOSE_FILE build --no-cache
        print_success "Build completed"
        ;;
        
    "clean")
        print_status "Cleaning up..."
        docker-compose -f $COMPOSE_FILE down -v --rmi all
        docker system prune -f
        print_success "Cleanup completed"
        ;;
        
    *)
        print_error "Unknown action: $ACTION"
        echo ""
        echo "Usage: $0 [compose-file] [action]"
        echo ""
        echo "Actions:"
        echo "  up/start   - Start services (default)"
        echo "  down/stop  - Stop services"
        echo "  restart    - Restart services"
        echo "  logs       - Show logs"
        echo "  ps/status  - Show service status"
        echo "  build      - Build services"
        echo "  clean      - Clean up everything"
        exit 1
        ;;
esac