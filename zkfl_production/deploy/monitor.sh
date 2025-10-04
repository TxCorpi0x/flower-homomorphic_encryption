#!/bin/bash

# Monitoring and management script for ZKFL production environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE:-"auto"}  # auto, docker, k8s
NAMESPACE="zkfl-system"
REFRESH_INTERVAL=5

print_header() {
    echo -e "${CYAN}$1${NC}"
}

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

# Detect deployment type
detect_deployment() {
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        echo "k8s"
    elif docker-compose -f zkfl_production/docker-compose.yml ps | grep -q "zkfl"; then
        echo "docker"
    else
        echo "none"
    fi
}

# Get deployment type
get_deployment_type() {
    if [ "$DEPLOYMENT_TYPE" = "auto" ]; then
        detect_deployment
    else
        echo $DEPLOYMENT_TYPE
    fi
}

# Monitor Docker deployment
monitor_docker() {
    clear
    print_header "🐳 ZKFL Docker Monitoring"
    echo "$(date)"
    echo ""
    
    print_info "Service Status:"
    docker-compose -f zkfl_production/docker-compose.yml ps
    echo ""
    
    print_info "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
    echo ""
    
    print_info "Recent Logs (Server):"
    docker-compose -f zkfl_production/docker-compose.yml logs --tail=5 zkfl-server 2>/dev/null || echo "No logs available"
    echo ""
    
    print_info "Recent Logs (Clients):"
    docker-compose -f zkfl_production/docker-compose.yml logs --tail=3 zkfl-client-1 2>/dev/null || echo "No logs available"
}

# Monitor Kubernetes deployment
monitor_k8s() {
    clear
    print_header "☸️  ZKFL Kubernetes Monitoring"
    echo "$(date)"
    echo ""
    
    print_info "Deployment Status:"
    kubectl get deployments -n $NAMESPACE 2>/dev/null || echo "No deployments found"
    echo ""
    
    print_info "Pod Status:"
    kubectl get pods -n $NAMESPACE 2>/dev/null || echo "No pods found"
    echo ""
    
    print_info "Service Status:"
    kubectl get services -n $NAMESPACE 2>/dev/null || echo "No services found"
    echo ""
    
    print_info "Resource Usage:"
    kubectl top pods -n $NAMESPACE 2>/dev/null || echo "Metrics not available (metrics-server required)"
    echo ""
    
    print_info "Recent Events:"
    kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' 2>/dev/null | tail -5 || echo "No events found"
}

# Show system health
show_health() {
    DEPLOYMENT=$(get_deployment_type)
    
    case $DEPLOYMENT in
        "docker")
            # Check Docker health
            if docker-compose -f zkfl_production/docker-compose.yml ps | grep -q "Up"; then
                print_success "Docker deployment is running"
                
                # Check server health
                if curl -s http://localhost:8080/health > /dev/null 2>&1; then
                    print_success "Server health check passed"
                else
                    print_error "Server health check failed"
                fi
            else
                print_error "Docker deployment is not running"
            fi
            ;;
            
        "k8s")
            # Check Kubernetes health
            if kubectl get deployment zkfl-server -n $NAMESPACE &> /dev/null; then
                READY=$(kubectl get deployment zkfl-server -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
                DESIRED=$(kubectl get deployment zkfl-server -n $NAMESPACE -o jsonpath='{.spec.replicas}')
                
                if [ "$READY" = "$DESIRED" ]; then
                    print_success "Kubernetes deployment is healthy ($READY/$DESIRED ready)"
                else
                    print_error "Kubernetes deployment is unhealthy ($READY/$DESIRED ready)"
                fi
                
                # Check service endpoint
                kubectl port-forward svc/zkfl-server-service 18080:8080 -n $NAMESPACE &> /dev/null &
                PF_PID=$!
                sleep 2
                
                if curl -s http://localhost:18080/health > /dev/null 2>&1; then
                    print_success "Server health check passed"
                else
                    print_error "Server health check failed"
                fi
                
                kill $PF_PID 2>/dev/null || true
            else
                print_error "Kubernetes deployment not found"
            fi
            ;;
            
        "none")
            print_error "No ZKFL deployment detected"
            ;;
    esac
}

# Show logs
show_logs() {
    DEPLOYMENT=$(get_deployment_type)
    SERVICE=${1:-"all"}
    
    case $DEPLOYMENT in
        "docker")
            case $SERVICE in
                "server")
                    docker-compose -f zkfl_production/docker-compose.yml logs -f zkfl-server
                    ;;
                "client"|"clients")
                    docker-compose -f zkfl_production/docker-compose.yml logs -f zkfl-client-1 zkfl-client-2 zkfl-client-3
                    ;;
                "all")
                    docker-compose -f zkfl_production/docker-compose.yml logs -f
                    ;;
            esac
            ;;
            
        "k8s")
            case $SERVICE in
                "server")
                    kubectl logs -f deployment/zkfl-server -n $NAMESPACE
                    ;;
                "client"|"clients")
                    kubectl logs -f deployment/zkfl-clients -n $NAMESPACE
                    ;;
                "all")
                    kubectl logs -f deployment/zkfl-server -n $NAMESPACE &
                    kubectl logs -f deployment/zkfl-clients -n $NAMESPACE &
                    wait
                    ;;
            esac
            ;;
            
        "none")
            print_error "No deployment found to show logs"
            ;;
    esac
}

# Interactive monitoring
monitor_interactive() {
    DEPLOYMENT=$(get_deployment_type)
    
    if [ "$DEPLOYMENT" = "none" ]; then
        print_error "No ZKFL deployment detected"
        exit 1
    fi
    
    print_info "Starting interactive monitoring (Press Ctrl+C to exit)"
    print_info "Detected deployment type: $DEPLOYMENT"
    echo ""
    
    while true; do
        case $DEPLOYMENT in
            "docker")
                monitor_docker
                ;;
            "k8s")
                monitor_k8s
                ;;
        esac
        
        echo ""
        print_info "Refreshing in $REFRESH_INTERVAL seconds... (Ctrl+C to exit)"
        sleep $REFRESH_INTERVAL
    done
}

# Main script
ACTION=${1:-"monitor"}

case $ACTION in
    "monitor"|"watch")
        monitor_interactive
        ;;
        
    "health"|"status")
        show_health
        ;;
        
    "logs")
        show_logs $2
        ;;
        
    "dashboard")
        DEPLOYMENT=$(get_deployment_type)
        case $DEPLOYMENT in
            "docker")
                print_info "Opening Docker dashboard..."
                if command -v docker-compose &> /dev/null; then
                    echo "Docker Compose services:"
                    docker-compose -f zkfl_production/docker-compose.yml ps
                    echo ""
                    echo "Access points:"
                    echo "  • Server: http://localhost:8080"
                    echo "  • Health: http://localhost:8080/health"
                fi
                ;;
            "k8s")
                print_info "Kubernetes dashboard access:"
                echo "  • Port forward: kubectl port-forward svc/zkfl-server-service 8080:8080 -n $NAMESPACE"
                echo "  • Dashboard: kubectl proxy (if dashboard is installed)"
                ;;
            "none")
                print_error "No deployment found"
                ;;
        esac
        ;;
        
    "info")
        DEPLOYMENT=$(get_deployment_type)
        print_header "📊 ZKFL Deployment Information"
        echo "Deployment type: $DEPLOYMENT"
        echo "Namespace: $NAMESPACE"
        echo "Timestamp: $(date)"
        echo ""
        
        show_health
        ;;
        
    *)
        print_error "Unknown action: $ACTION"
        echo ""
        echo "Usage: $0 [action]"
        echo ""
        echo "Actions:"
        echo "  monitor/watch    - Interactive monitoring (default)"
        echo "  health/status    - Show health status"
        echo "  logs [service]   - Show logs (server|client|all)"
        echo "  dashboard        - Show dashboard access info"
        echo "  info             - Show deployment information"
        echo ""
        echo "Environment variables:"
        echo "  DEPLOYMENT_TYPE  - Force deployment type (docker|k8s|auto)"
        echo "  REFRESH_INTERVAL - Monitoring refresh interval (default: 5)"
        exit 1
        ;;
esac