# 🚀 ZK + Blockchain Enhanced Federated Learning

A cutting-edge federated learning system that replaces traditional Homomorphic Encryption (FHE) with **Zero-Knowledge Proofs**, **Differential Privacy**, and **Blockchain** for enhanced security, transparency, and verifiability.

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [🔒 Security Architecture](#-security-architecture)
- [🏗️ System Architecture](#️-system-architecture)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📖 Usage Examples](#-usage-examples)
- [🔧 Configuration](#-configuration)
- [🧪 Testing](#-testing)
- [📊 Architecture Comparison](#-architecture-comparison)
- [🛠️ Development](#️-development)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🌟 Overview

This project implements a next-generation federated learning system that addresses the limitations of traditional cryptographic approaches by introducing:

- **🔐 Zero-Knowledge Proofs**: Cryptographic verification of training integrity without revealing model weights
- **🛡️ Differential Privacy**: Formal privacy guarantees with configurable noise injection
- **⛓️ Blockchain Integration**: Transparent and immutable audit trail for all training operations
- **🚀 Enhanced Performance**: Elimination of FHE computational overhead while maintaining security

### Key Benefits

- **Stronger Security**: Multi-layered protection with ZK proofs, DP, and blockchain
- **Complete Transparency**: All training operations recorded on blockchain
- **Cryptographic Verifiability**: No trust assumptions required
- **Regulatory Compliance**: Built-in auditability for compliance requirements
- **Production Ready**: Mock implementations can be replaced with production ZK systems

## 🔒 Security Architecture

### 🔐 Zero-Knowledge Proofs
- **Purpose**: Verify training integrity without revealing sensitive information
- **Implementation**: Circuit-based proof system with commitment schemes
- **Benefits**: Cryptographic guarantees of honest training

### 🛡️ Differential Privacy
- **Purpose**: Protect individual data contributions with formal privacy guarantees
- **Implementation**: Gaussian noise injection with gradient clipping
- **Configuration**: Adjustable noise scale, clipping norm, and privacy budget

### ⛓️ Blockchain Integration
- **Purpose**: Immutable audit trail and transparent coordination
- **Implementation**: Smart contracts for FL coordination and proof verification
- **Benefits**: Complete transparency and regulatory compliance

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client 1      │    │   FL Server      │    │   Blockchain    │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │   Model     │ │    │ │  Aggregator  │ │    │ │ Smart       │ │
│ │  Training   │ │    │ │   Strategy   │ │    │ │ Contract    │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│        │        │    │        │         │    │        │        │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ ZK Proof    │◄├────┼─┤ ZK Verifier  │ │    │ │  Proof      │ │
│ │ Generator   │ │    │ │              │ │    │ │  Storage    │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│        │        │    │        │         │    │        │        │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │   DP        │ │    │ │  Blockchain  │◄├────┼─┤   Audit     │ │
│ │ Protection  │ │    │ │  Interface   │ │    │ │   Trail     │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    📊 Secure Aggregation
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- conda (recommended) or pip
- Git

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TxCorpi0x/flower-homomorphic_encryption.git
   cd flower-homomorphic_encryption
   ```

2. **Create conda environment**:
   ```bash
   conda create -n flEnv python=3.11
   conda activate flEnv
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Blockchain Setup (Optional)

For testing with actual blockchain:

1. **Install Ganache CLI**:
   ```bash
   npm install -g ganache-cli
   ```

2. **Start local blockchain**:
   ```bash
   ganache-cli --deterministic --accounts 10 --host 0.0.0.0
   ```

## 🚀 Quick Start

### 1. Run Architecture Comparison

Compare FHE vs ZK+Blockchain approaches:

```bash
python zkfl_example.py --compare
```

### 2. Run Comprehensive Demos

Experience different federated learning approaches:

```bash
# Centralized learning (traditional ML)
python zkfl_example.py --approach centralized

# Classic federated learning
python zkfl_example.py --approach classic

# FL with Zero-Knowledge proofs
python zkfl_example.py --approach zk

# FL with Differential Privacy
python zkfl_example.py --approach dp

# Hybrid approach (ZK + DP + Blockchain)
python zkfl_example.py --approach hybrid
```

### 3. Run Tests

Verify the implementation:

```bash
python test_zk_blockchain.py
```

### 4. Run Production Federated Learning

Deploy actual distributed federated learning:

**Start Server**:
```bash
python zkfl_production/run_server.py --enable_zk --enable_blockchain
```

**Start Clients** (in separate terminals):
```bash
python zkfl_production/run_client.py 0 --enable_zk --differential_privacy
python zkfl_production/run_client.py 1 --enable_zk --differential_privacy
python zkfl_production/run_client.py 2 --enable_zk --differential_privacy
```

## 📖 Usage Examples

### Basic Federated Learning with ZK Proofs

```python
from zkfl.crypto.zk_proof import ZKProofSystem
from zkfl.privacy.differential_privacy import DifferentialPrivacyManager

# Initialize security components
zk_prover = ZKProofSystem()
dp_system = DifferentialPrivacyManager(epsilon=1.0, delta=1e-5)

# Generate ZK proof for model update
proof = zk_prover.generate_training_proof(
    old_model_params=old_params,
    new_model_params=new_params,
    training_data_hash="data_hash",
    learning_rate=0.01,
    epochs=1,
    client_id="client_1"
)

# Verify proof
is_valid = zk_prover.verify_proof(proof)
print(f"Proof valid: {is_valid}")
```

### Blockchain Integration

```python
from zkfl.blockchain.blockchain_interface import BlockchainInterface

# Initialize blockchain connection
blockchain = BlockchainInterface(provider_url="http://localhost:8545")

# Submit round results with ZK proof
tx_hash = blockchain.submit_round_results(
    round_number=1,
    round_metrics={"accuracy": 0.85, "loss": 0.15},
    client_updates=[{
        "client_id": "client_1",
        "proof": proof_data,
        "model_commitment": "commitment_hash"
    }]
)

# Verify model integrity
is_verified = blockchain.verify_model_integrity("commitment_hash")
```

### Production Deployment

```python
# Start production server
from zkfl_production import ZKFLProductionServer

server = ZKFLProductionServer(config_path="production_config.yaml")
server.run()

# Start production client
from zkfl_production import ZKFLProductionClient

client = ZKFLProductionClient(client_id=0, config_path="production_config.yaml")
client.run()
```

## 🔧 Configuration

### ZK Proof Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--enable_zk` | Enable zero-knowledge proofs | `False` |
| `--zk_circuit_path` | Path to ZK circuit definition | `./circuits/fl_circuit.json` |
| `--zk_proving_key` | Path to proving key | `./keys/proving_key.json` |
| `--proof_save_path` | Directory for proof storage | `./proofs` |

### Differential Privacy Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--differential_privacy` | Enable differential privacy | `False` |
| `--noise_scale` | DP noise scale (higher = more privacy) | `1.0` |
| `--clip_norm` | Gradient clipping norm | `1.0` |
| `--privacy_budget` | Total privacy budget (epsilon) | `1.0` |

### Blockchain Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--enable_blockchain` | Enable blockchain logging | `False` |
| `--blockchain_provider` | Blockchain RPC URL | `http://localhost:8545` |
| `--contract_address` | Smart contract address | `None` |
| `--private_key` | Private key for transactions | `None` |

### Training Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--rounds` | Number of FL rounds | `3` |
| `--number_clients` | Number of clients | `3` |
| `--max_epochs` | Local training epochs | `1` |
| `--batch_size` | Training batch size | `64` |
| `--lr` | Learning rate | `0.001` |

## 🧪 Testing

### Comprehensive Test Suite

Run all tests:
```bash
python test_zk_blockchain.py
```

### Individual Component Tests

**ZK Proof System**:
```python
from going_modular.zk_security import ZKModelProof
zk = ZKModelProof()
# Test proof generation and verification
```

**Differential Privacy**:
```python
from going_modular.zk_security import DifferentialPrivacy
dp = DifferentialPrivacy(noise_scale=0.1)
# Test noise injection
```

**Blockchain Interface**:
```python
from going_modular.zk_security import BlockchainStorage
blockchain = BlockchainStorage()
# Test blockchain operations
```

### Performance Benchmarks

```bash
# Benchmark ZK proof generation
python -m pytest tests/test_zk_performance.py

# Benchmark differential privacy overhead
python -m pytest tests/test_dp_performance.py

# Benchmark blockchain operations
python -m pytest tests/test_blockchain_performance.py
```

## 📊 Architecture Comparison

| **Aspect** | **Original (FHE)** | **New (ZK + Blockchain)** |
|------------|---------------------|----------------------------|
| **Privacy Method** | Homomorphic Encryption | Differential Privacy |
| **Integrity Verification** | Limited | Zero-Knowledge Proofs |
| **Transparency** | None | Blockchain Audit Trail |
| **Performance** | High computational cost | Efficient operations |
| **Verifiability** | Trust-based | Cryptographically proven |
| **Auditability** | Not available | Immutable blockchain records |
| **Scalability** | Limited by FHE overhead | Highly scalable |
| **Regulatory Compliance** | Difficult | Built-in compliance features |

## 🛠️ Development

### Project Structure

```
flower-homomorphic_encryption/
├── zkfl/                   # Core ZKFL framework
│   ├── crypto/            # ZK proofs and cryptographic utilities
│   ├── privacy/           # Differential privacy implementation
│   ├── blockchain/        # Blockchain integration
│   ├── federated/         # Flower FL integration
│   ├── models/            # Neural network models
│   ├── data/              # Data loading utilities
│   ├── utils/             # Configuration and logging
│   └── core/              # Core training engine
├── zkfl_production/       # Production FL deployment
│   ├── server.py          # Production FL server
│   ├── client.py          # Production FL client
│   ├── run_server.py      # Server entry point
│   └── run_client.py      # Client entry point
├── circuits/              # ZK circuit definitions
├── keys/                  # Cryptographic keys
├── proofs/                # Generated ZK proofs
├── contracts/             # Smart contracts
├── zkfl_example.py        # Comprehensive demo script
├── test_zk_blockchain.py  # Test suite
├── requirements.txt       # Dependencies
└── README.md              # This file
```

### Adding New ZK Circuits

1. **Define circuit**: Create circuit definition in `circuits/`
2. **Generate keys**: Generate proving/verifying keys
3. **Update config**: Update ZK prover configuration
4. **Test**: Verify circuit works with test cases

### Custom Smart Contracts

1. **Write contract**: Create Solidity contract in `contracts/`
2. **Compile**: Use `py-solc-x` for compilation
3. **Deploy**: Deploy to blockchain network
4. **Update interface**: Update blockchain storage interface

### Extending Differential Privacy

1. **Custom mechanisms**: Implement in `zk_security.py`
2. **New algorithms**: Add to `DifferentialPrivacy` class
3. **Privacy analysis**: Update privacy accounting
4. **Validation**: Add comprehensive tests

## 📚 Legacy FHE Documentation

> **Note**: The following sections document the original FHE implementation that has been replaced with ZK + Blockchain architecture. Kept for reference purposes.

<details>
<summary>Click to expand legacy FHE documentation</summary>

### Original FHE Configuration

The original system used TenSEAL for homomorphic encryption:

```bash
# Legacy FHE training
python simulation.py simulation --he --data_path data/ --dataset cifar
```

#### Creating FHE Keys (Legacy)

```bash
python create_keys.py
```

#### FHE Parameters (Legacy)

- `--he`: Enable homomorphic encryption (replaced with `--enable_zk`)
- `--path_crypted`: Path for encrypted results (replaced with blockchain storage)
- `--path_public_key`: Public key path (replaced with ZK proving keys)
- `--path_keys`: Private/public key combination (replaced with ZK verification)

</details>

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Install dev dependencies: `pip install -r requirements.txt`
4. Make changes and add tests
5. Run test suite: `python test_zk_blockchain.py`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push branch: `git push origin feature/amazing-feature`
8. Open Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 🔒 Security Considerations

### Production Deployment

- **Replace mock ZK implementations** with production libraries (Circom, arkworks, etc.)
- **Use hardware security modules** (HSMs) for key management
- **Implement proper key rotation** for blockchain accounts
- **Audit smart contracts** before mainnet deployment
- **Monitor privacy budget** consumption in production

### Known Limitations

- **Mock ZK proofs**: Current implementation uses mock proofs for demonstration
- **Local blockchain**: Default configuration uses local test blockchain
- **Development keys**: Includes development-only cryptographic keys

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Flower Team**: For the excellent federated learning framework
- **ZK Community**: For zero-knowledge proof research and tools
- **Blockchain Developers**: For Web3 integration libraries
- **Privacy Researchers**: For differential privacy implementations

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/TxCorpi0x/flower-homomorphic_encryption/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TxCorpi0x/flower-homomorphic_encryption/discussions)
- **Documentation**: [Wiki](https://github.com/TxCorpi0x/flower-homomorphic_encryption/wiki)

---

**Built with ❤️ for secure, transparent, and verifiable federated learning**
