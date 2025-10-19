# Federated Learning with Privacy-Preserving Techniques

## Complete Implementation Summary

This project implements **four** privacy-preserving approaches for federated learning:

1. **Baseline**: Standard FL (no privacy protection)
2. **HE (Homomorphic Encryption)**: TenSEAL CKKS encryption
3. **ZKP (Zero-Knowledge Proofs)**: Pedersen commitments
4. **DP (Differential Privacy)**: Gaussian noise with gradient clipping

---

## Quick Start

### 1. Setup Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Create cryptographic parameters
python create_keys.py          # For HE mode
python create_zkp_params.py    # For ZKP mode
python create_dp_params.py     # For DP mode
```

### 2. Run Individual Modes

**Baseline (fastest):**
```bash
python simulation.py simulation --benchmark --rounds 2 --number_clients 2
```

**HE (encrypted parameters):**
```bash
python simulation.py simulation --he --benchmark --rounds 2 --number_clients 2
```

**ZKP (parameter commitments):**
```bash
python simulation.py simulation --zkp --zkp_params zkp_params.pkl --benchmark --rounds 2 --number_clients 2
```

**DP (noisy parameters):**
```bash
python simulation.py simulation --dp --dp_params dp_params.pkl --benchmark --rounds 2 --number_clients 2
```

### 3. Compare All Modes

```bash
python compare_methods_simple.py --modes baseline,he,zkp,dp --rounds 2 --number_clients 2
```

---

## Performance Comparison

### Expected Results (2 clients, 1 round, CIFAR-10, CPU):

| Mode | Total Time | Crypto Overhead | Communication | Accuracy | Privacy Level |
|------|------------|-----------------|---------------|----------|---------------|
| **Baseline** | ~10s | 0s | 0.47MB | 32% | None |
| **HE** | ~10s | ~0.6s | 0.47MB | 27-34% | Very Strong |
| **ZKP** | ~80s | ~70s | 0.47MB | 29-32% | Strong |
| **DP** | ~10s | ~0.01s | 0.47MB | 25-32% | Tunable |

---

## Privacy-Performance Tradeoff

### Privacy Strength:
```
HE > ZKP > DP (ε<1) > DP (ε>1) > Baseline
```

### Computational Overhead:
```
ZKP >> HE > DP ≈ Baseline
```

### Use Case Recommendations:

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| **Maximum privacy, any cost** | HE | Full encryption |
| **Balance privacy & speed** | DP or ZKP | Tunable overhead |
| **Regulatory compliance** | DP | Formal guarantees |
| **Untrusted server** | HE | Server sees encrypted data |
| **Limited compute** | DP | Minimal overhead |
| **Proof of correctness** | ZKP | Verify without revealing |
| **Maximum speed** | Baseline | No protection |

---

## Architecture Overview

### Model:
- **LeNet-5 style CNN** (~62K parameters)
- 2 Conv layers + 3 FC layers
- Trained on CIFAR-10 (10 classes)

### Framework:
- **Flower 1.18.0** for federated learning
- **TenSEAL** for homomorphic encryption
- **Crypto libraries** for ZKP
- **NumPy** for DP noise

### Benchmarking:
- Comprehensive timing metrics
- Memory usage tracking
- Communication overhead
- Model quality metrics
- Crypto operation statistics

---

## File Structure

```
├── core/
│   ├── model_builder.py    # CNN model definition
│   ├── security.py          # HE + DP implementations
│   ├── zkp.py               # ZKP implementation
│   └── benchmark.py         # Performance tracking
├── client.py                # FL client logic
├── server.py                # FL server strategy
├── simulation.py            # Simulation mode entry
├── create_keys.py           # Generate HE keys
├── create_zkp_params.py     # Generate ZKP params
├── create_dp_params.py      # Generate DP params
├── compare_methods_simple.py# Compare all modes
└── results/                 # Output directory
```

---

## Detailed Guides

- **[HE Guide](./README.md)** - Homomorphic Encryption setup
- **[ZKP Guide](./ZKP_GUIDE.md)** - Zero-Knowledge Proofs usage
- **[DP Guide](./DP_GUIDE.md)** - Differential Privacy configuration
- **[Comparison Guide](./COMPARISON_GUIDE.md)** - Benchmark analysis
- **[Benchmark Guide](./BENCHMARK_COMPARISON.md)** - Performance metrics

---

## Key Features

✅ **Four privacy modes**: Baseline, HE, ZKP, DP
✅ **Comprehensive benchmarking**: Timing, memory, communication, accuracy
✅ **Simulation & federated**: Both execution modes supported
✅ **Configurable parameters**: Customize privacy-utility tradeoff
✅ **Visualization**: Automatic comparison plots
✅ **Production-ready**: Error handling, logging, documentation

---

## Privacy Comparison

### Homomorphic Encryption (HE)
- **Privacy**: Cryptographic encryption (IND-CPA secure)
- **Mechanism**: Operations on encrypted data
- **Advantage**: Server never sees plaintext
- **Disadvantage**: ~0.6s overhead per round
- **Best for**: Untrusted servers, maximum privacy

### Zero-Knowledge Proofs (ZKP)
- **Privacy**: Computational hiding (discrete log problem)
- **Mechanism**: Commitments + proofs of correctness
- **Advantage**: Verifiable correctness
- **Disadvantage**: ~70s overhead per round
- **Best for**: Audit requirements, proof-based trust

### Differential Privacy (DP)
- **Privacy**: Information-theoretic (ε-DP guarantee)
- **Mechanism**: Calibrated noise addition
- **Advantage**: ~0.01s overhead, formal guarantees
- **Disadvantage**: Accuracy loss (depends on ε)
- **Best for**: Regulatory compliance, tunable privacy

---

## Example Workflows

### Research Experiment:
```bash
# Compare all modes with detailed benchmarking
python compare_methods_simple.py \
    --modes baseline,he,zkp,dp \
    --rounds 5 \
    --number_clients 4 \
    --max_epochs 2 \
    --batch_size 32

# Results in ./results/comparison_all_modes/
# - comparison.png (visualizations)
# - comparison_report.json (detailed metrics)
# - baseline/, he/, zkp/, dp/ (individual results)
```

### Production Deployment (DP):
```bash
# 1. Tune DP parameters for your privacy requirements
python create_dp_params.py --epsilon 1.0 --delta 1e-5

# 2. Run federated training
python main_server.py server --dp --rounds 10 --benchmark

# Clients (on different machines):
python main_client.py client --dp --dp_params dp_params.pkl --id_client 0
python main_client.py client --dp --dp_params dp_params.pkl --id_client 1
```

### High-Security Deployment (HE):
```bash
# 1. Generate encryption keys (once)
python create_keys.py

# 2. Distribute public key to server, private keys to clients

# 3. Run federated training with encryption
python main_server.py server --he --rounds 10 --benchmark
python main_client.py client --he --id_client 0
```

---

## Citation

If you use this code for research, please cite:

```bibtex
@software{fl_privacy_comparison_2025,
  title = {Federated Learning with Privacy-Preserving Techniques},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/TxCorpi0x/flower-homomorphic_encryption}
}
```

---

## License

[Your License Here]

---

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Secure aggregation protocols
- [ ] More DP mechanisms (Rényi DP, zCDP)
- [ ] Advanced composition accounting
- [ ] Additional ZKP schemes (Schnorr, Bulletproofs)
- [ ] Model compression techniques
- [ ] Byzantine-robust aggregation
- [ ] Personalized federated learning
- [ ] Cross-device FL scenarios

---

## Support

For questions or issues:
- Open a GitHub issue
- Check the guides: `*_GUIDE.md` files
- Review benchmarks: `results/` directory

---

**✨ Complete privacy-preserving federated learning framework with 4 modes, comprehensive benchmarking, and production-ready code!**
