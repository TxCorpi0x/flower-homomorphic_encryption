# 🚀 ZKFL Production Environment

A complete production-ready deployment system for Zero-Knowledge Federated Learning (ZKFL) with support for Docker, Kubernetes, and cloud platforms.

## 🏗️ Architecture

```
zkfl_production/
├── 📁 deploy/           # Deployment scripts
│   ├── build.sh         # Build Docker images
│   ├── deploy-docker.sh # Docker deployment
│   ├── deploy-k8s.sh    # Kubernetes deployment
│   └── monitor.sh       # Monitoring & management
├── 📁 k8s/              # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── server-deployment.yaml
│   └── client-deployment.yaml
├── 📁 configs/          # Configuration templates
│   ├── .env.template    # Production config template
│   └── .env.dev         # Development config
├── 🐳 Dockerfile        # Base image
├── 🐳 Dockerfile.server # Server-specific image
├── 🐳 Dockerfile.client # Client-specific image
├── 🐳 docker-compose.yml      # Production compose
├── 🐳 docker-compose.dev.yml  # Development compose
├── 🔧 zkfl_production_server.py # Production server
├── 🔧 zkfl_production_client.py # Production client
├── ▶️ run_server.py     # Server entry point
└── ▶️ run_client.py     # Client entry point
```

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended for local development)

```bash
# 1. Build images
./zkfl_production/deploy/build.sh

# 2. Start development environment
./zkfl_production/deploy/deploy-docker.sh zkfl_production/docker-compose.dev.yml up

# 3. Start production environment
./zkfl_production/deploy/deploy-docker.sh zkfl_production/docker-compose.yml up

# 4. Monitor services
./zkfl_production/deploy/monitor.sh
```

### Option 2: Kubernetes (For production clusters)

```bash
# 1. Build and push images
REGISTRY=your-registry.com PUSH=true ./zkfl_production/deploy/build.sh

# 2. Deploy to Kubernetes
REGISTRY=your-registry.com ./zkfl_production/deploy/deploy-k8s.sh deploy

# 3. Monitor deployment
./zkfl_production/deploy/monitor.sh

# 4. Access services
kubectl port-forward svc/zkfl-server-service 8080:8080 -n zkfl-system
```

### Option 3: Manual/Development (Traditional approach)

```bash
# 1. Start server
python zkfl_production/run_server.py --enable_zk --enable_blockchain

# 2. Start clients (in separate terminals)
python zkfl_production/run_client.py 0 --enable_zk --differential_privacy
python zkfl_production/run_client.py 1 --enable_zk --differential_privacy
python zkfl_production/run_client.py 2 --enable_zk --differential_privacy
```

## 🔧 Configuration

### Environment Variables

Copy and customize the configuration:
```bash
cp zkfl_production/configs/.env.template zkfl_production/.env
```

Key settings:
- `NUM_ROUNDS`: Number of federated learning rounds
- `MIN_NUM_CLIENTS`: Minimum clients required
- `ZK_ENABLED`: Enable zero-knowledge proofs
- `DP_ENABLED`: Enable differential privacy
- `BLOCKCHAIN_ENABLED`: Enable blockchain integration

### Security Features

- **Zero-Knowledge Proofs**: Privacy-preserving model updates
- **Differential Privacy**: Noise-based privacy protection
- **TLS/HTTPS**: Encrypted communication
- **Blockchain Integration**: Immutable audit trail (optional)

## 📊 Monitoring & Management

### Real-time Monitoring
```bash
# Interactive dashboard
./zkfl_production/deploy/monitor.sh monitor

# Health check
./zkfl_production/deploy/monitor.sh health

# View logs
./zkfl_production/deploy/monitor.sh logs server
./zkfl_production/deploy/monitor.sh logs clients
```

### Service Management

#### Docker
```bash
# Start services
./zkfl_production/deploy/deploy-docker.sh zkfl_production/docker-compose.yml up

# Stop services  
./zkfl_production/deploy/deploy-docker.sh zkfl_production/docker-compose.yml down

# View status
./zkfl_production/deploy/deploy-docker.sh zkfl_production/docker-compose.yml status

# Scale clients
docker-compose -f zkfl_production/docker-compose.yml up --scale zkfl-client-1=5
```

#### Kubernetes
```bash
# Deploy
./zkfl_production/deploy/deploy-k8s.sh deploy

# Scale clients
./zkfl_production/deploy/deploy-k8s.sh scale 5

# Delete deployment
./zkfl_production/deploy/deploy-k8s.sh delete

# Port forwarding
./zkfl_production/deploy/deploy-k8s.sh port-forward 8080
```

## 🌐 Cloud Deployment

### AWS EKS
```bash
# Create EKS cluster
eksctl create cluster --name zkfl-cluster --region us-west-2

# Deploy ZKFL
REGISTRY=your-account.dkr.ecr.us-west-2.amazonaws.com ./zkfl_production/deploy/deploy-k8s.sh deploy
```

### Google GKE
```bash
# Create GKE cluster
gcloud container clusters create zkfl-cluster --zone us-central1-a

# Deploy ZKFL
REGISTRY=gcr.io/your-project-id ./zkfl_production/deploy/deploy-k8s.sh deploy
```

### Azure AKS
```bash
# Create AKS cluster
az aks create --resource-group zkfl-rg --name zkfl-cluster

# Deploy ZKFL
REGISTRY=yourregistry.azurecr.io ./zkfl_production/deploy/deploy-k8s.sh deploy
```

## 🔍 API Endpoints

### Server Endpoints
- `GET /health` - Health check
- `GET /ready` - Readiness probe
- `GET /metrics` - Prometheus metrics (if enabled)
- `POST /start` - Start federated learning
- `GET /status` - Current training status

### Client Registration
Clients automatically register with the server using Flower's gRPC protocol.

## 🧪 Testing Production Setup

### Load Testing
```bash
# Start with minimal configuration
NUM_ROUNDS=2 MIN_NUM_CLIENTS=1 ./zkfl_production/deploy/deploy-docker.sh zkfl_production/docker-compose.dev.yml up

# Scale up gradually
docker-compose -f zkfl_production/docker-compose.yml up --scale zkfl-client-1=10
```

### Performance Testing
```bash
# Enable monitoring
./zkfl_production/deploy/monitor.sh monitor

# Check resource usage
docker stats
# or for Kubernetes
kubectl top pods -n zkfl-system
```

## 🔒 Security Considerations

### Production Checklist
- [ ] Enable TLS/HTTPS (`HTTPS_PORT=8443`)
- [ ] Configure secure key management
- [ ] Enable audit logging
- [ ] Set up network policies (Kubernetes)
- [ ] Configure resource limits
- [ ] Enable differential privacy
- [ ] Rotate encryption keys regularly

### Key Management
```bash
# Generate cryptographic keys
mkdir -p zkfl_production/keys
# Keys are auto-generated on first startup
```

## 🔧 Troubleshooting

### Common Issues

#### Services not starting
```bash
# Check logs
./zkfl_production/deploy/monitor.sh logs all

# Check configuration
./zkfl_production/deploy/monitor.sh health
```

#### Clients not connecting
```bash
# Verify network connectivity
docker-compose -f zkfl_production/docker-compose.yml exec zkfl-client-1 ping zkfl-server

# Check server logs
./zkfl_production/deploy/monitor.sh logs server
```

#### Performance issues
```bash
# Monitor resource usage
./zkfl_production/deploy/monitor.sh monitor

# Scale clients
./zkfl_production/deploy/deploy-k8s.sh scale 2  # Reduce load
```

### Debug Mode
```bash
# Enable debug logging
DEBUG=true ./zkfl_production/deploy/deploy-docker.sh zkfl_production/docker-compose.dev.yml up
```

## 📚 Architecture Comparison

| **Aspect** | **`zkfl_example.py`** | **`zkfl_production/`** |
|------------|----------------------|------------------------|
| **Purpose** | Demo & Testing | Production Deployment |
| **Architecture** | Single-process simulation | Multi-process distributed FL |
| **Flower Integration** | ❌ No | ✅ Yes |
| **Real Federation** | ❌ Simulated | ✅ Actual client-server |
| **Network Communication** | ❌ Local only | ✅ Network protocols |
| **Multi-client Support** | ❌ Mock | ✅ Real concurrent clients |
| **Docker Support** | ❌ No | ✅ Full containerization |
| **Kubernetes Support** | ❌ No | ✅ Cloud-ready manifests |
| **Monitoring** | ❌ Basic logs | ✅ Full observability |
| **Scalability** | ❌ Limited | ✅ Auto-scaling |

## 🚀 Next Steps

1. **Customize Configuration**: Adjust `zkfl_production/.env` for your use case
2. **Scale Testing**: Test with multiple clients
3. **Security Hardening**: Enable all security features
4. **Monitoring Setup**: Integrate with Prometheus/Grafana
5. **CI/CD Integration**: Add automated deployments

## 📚 Additional Resources

- [ZKFL Core Documentation](../README.md)
- [Flower Framework](https://flower.dev/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Kubernetes](https://kubernetes.io/docs/)

---

For support and contributions, see the main [ZKFL repository](../).