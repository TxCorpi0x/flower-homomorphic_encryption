# Federated Learning Comparison Guide

## Overview

This project now supports comparing **four** privacy-preserving federated learning approaches:

1. **Baseline**: Standard federated learning (no privacy protection)
2. **Homomorphic Encryption (HE)**: Encrypts model weights using TenSEAL/CKKS with real encrypted transport
3. **Zero-Knowledge Proofs (ZKP)**: Proves model integrity using Pedersen commitments
4. **Differential Privacy (DP)**: Adds calibrated noise for formal privacy guarantees

## Quick Start

### 1. Setup (Docker - Recommended)

```bash
# Initialize cryptographic parameters
docker compose --profile init run --rm init

# Run all modes sequentially
bash scripts/run_docker_compare.sh

# Generate comparison report
docker compose --profile aggregate run --rm aggregate
```

### 2. Run Specific Modes

```bash
# Baseline
docker compose --profile baseline up --abort-on-container-exit

# Homomorphic Encryption
docker compose --profile he up --abort-on-container-exit

# Zero-Knowledge Proofs
docker compose --profile zkp up --abort-on-container-exit

# Differential Privacy
docker compose --profile dp up --abort-on-container-exit
```

### 3. View Results

After running, find results in `./results/docker_compare/`:
- `comparison.png` - Visual comparison plots (6 subplots)
- `comparison_report.json` - Detailed metrics
- `baseline/benchmark.json` - Baseline metrics
- `he_tenseal/benchmark.json` - HE metrics
- `zkp/benchmark.json` - ZKP metrics
- `dp/benchmark.json` - DP metrics

## Comparison Metrics

The comparison provides:

### Performance Metrics
- **Total Training Time**: End-to-end FL execution time
- **Client Fit Time**: Average time per client training round
- **Server Aggregate Time**: Time for parameter aggregation
- **Communication Overhead**: Upload/download data volume

### Privacy Metrics
- **Cryptographic Overhead**: Time spent on encryption/proofs/noise
- **Proof Generation**: Time to create ZKP proofs (ZKP mode)
- **Proof Verification**: Time to verify proofs (ZKP mode)
- **Encryption/Decryption**: Time for HE operations (HE mode)
- **DP Noise Addition**: Time for gradient noise (DP mode)

### Model Quality Metrics  
- **Training Accuracy**: Local model accuracy per client
- **Validation Accuracy**: Model accuracy on validation set
- **Global Model Accuracy**: Aggregated model performance
- **Loss Convergence**: Initial vs final model loss

## Available Comparison Methods

### 1. Docker Mode (Recommended)

**Best for: Full end-to-end comparison with all four modes**

Uses real gRPC transport, containerized execution, production-ready.

```bash
# Run all modes
bash scripts/run_docker_compare.sh

# Aggregate results
docker compose --profile aggregate run --rm aggregate
```

**Pros:**
- **All four modes work reliably** (baseline, he, zkp, dp)
- Real gRPC transport (no simulation)
- Containerized, reproducible environment
- Full benchmarking support
- Visual comparison plots
- **HE now works with real encrypted transport** (serialization fixed)

**Results:**
- Total time: ~500s for all four modes (3 rounds each)
- Comprehensive metrics for all privacy approaches
- Side-by-side comparison plots

### 2. Simulation Mode (Alternative)

**Best for: Quick local testing**

```bash
# Test individual modes
python simulation.py simulation --benchmark --rounds 2 --number_clients 2
python simulation.py simulation --he --benchmark --rounds 2 --number_clients 2
python simulation.py simulation --zkp --zkp_params zkp_params.pkl --benchmark --rounds 2
python simulation.py simulation --dp --dp_params dp_params.pkl --benchmark --rounds 2
```

**Pros:**
- Faster startup (no Docker overhead)
- Good for development/debugging

**Cons:**
- Simulation framework limitations
- May not reflect real gRPC behavior

## Understanding Results

### Typical Performance Profile (Docker, 3 rounds, 2 clients)

**Baseline** (Fast, no privacy):
- Total time: ~139s
- Crypto overhead: 0s
- Accuracy: ~45.79%
- Communication: ~1.4 MB

**HE** (Real encrypted transport):
- Total time: ~143s
- Crypto overhead: ~3.7s (encryption/decryption)
- Accuracy: ~45.29%
- Communication: ~328 MB (encrypted parameters larger)
- **Status**: ✅ Working with real TenSEAL serialization over gRPC

**ZKP** (Proof-based integrity):
- Total time: ~243s
- Crypto overhead: ~93s (proof generation + verification)
- Accuracy: ~47.00%
- Communication: ~1.4 MB (includes commitments)

**DP** (Noise-based privacy):
- Total time: ~151s
- Crypto overhead: <0.1s (noise addition)
- Accuracy: ~44.81%
- Communication: ~1.4 MB

### Performance Trade-offs

| Mode | Privacy Level | Speed | Communication | Accuracy Impact |
|------|--------------|-------|---------------|-----------------|
| Baseline | None | Fast | Low | Baseline (~46%) |
| HE | Highest (Confidentiality) | Medium | Very High | Similar (~45%) |
| ZKP | High (Integrity) | Slow | Low | Similar (~47%) |
| DP | Medium (Formal ε-DP) | Fast | Low | Tunable (~45%) |

## Troubleshooting

### Docker Issues

**Error:** Port already in use (8081-8084)

**Solution:** Change ports in `docker-compose.yml` or stop conflicting services

**Error:** TenSEAL build fails

**Solution:** The Dockerfile ensures `platform: linux/amd64` and installs build tools

### Key/Parameter Issues

**Error:** `FileNotFoundError: secret.pkl`

**Solution:**
```bash
docker compose --profile init run --rm init
```

Or locally:
```bash
python create_keys.py
python create_zkp_params.py
python create_dp_params.py
```

### HE Scale Issues

**Error:** "scale out of bounds" during aggregation

**Solution:** The current implementation uses normalized weights (num/total) to prevent scale overflow. This is already fixed in `server.py`.

## Configuration Options

### Docker Mode

```bash
# Run specific mode
docker compose --profile <mode> up --abort-on-container-exit
# where <mode> is: baseline, he, zkp, dp

# Configure via docker-compose.yml:
# - Number of rounds
# - Number of clients
# - Batch size
# - Device (cpu/cuda)
```

### Simulation Mode

```bash
python simulation.py simulation \
  --rounds 5 \                     # FL rounds
  --number_clients 4 \             # Number of clients
  --max_epochs 1 \                 # Local epochs per round
  --batch_size 64 \                # Training batch size
  --device cpu \                   # Device (cpu/cuda/mps)
  --benchmark                      # Enable benchmarking
```

## Best Practices

1. **Use Docker Mode**: Most reliable for full comparison (all four modes work)
2. **Start Small**: Test with 2 clients, 3 rounds first
3. **Monitor Resources**: Check `docker stats` for memory usage
4. **Check Logs**: Look at container logs with `docker compose logs <service>`
5. **Validate Benchmarks**: Ensure `benchmark.json` files exist with non-zero values
6. **Run Aggregation**: Always run the aggregate profile after all modes complete

## Implementation Status

- ✅ **Baseline Mode**: Fully working with benchmarking
- ✅ **HE Mode**: Fully working with real encrypted gRPC transport (scale issue fixed)
- ✅ **ZKP Mode**: Fully working with benchmarking  
- ✅ **DP Mode**: Fully working with benchmarking
- ✅ **Comparison Framework**: Working for all four modes
- ✅ **Visualization**: 6-plot comparison dashboard
- ✅ **Docker Support**: All modes containerized and tested

## Future Improvements

Potential enhancements:

1. **GPU Support**: Enable GPU acceleration in Docker
2. **More Datasets**: Support ImageNet, medical imaging datasets
3. **Secure Aggregation**: Implement multi-party secure aggregation
4. **Byzantine Robustness**: Add robust aggregation against malicious clients
5. **Cross-Device FL**: Support mobile/edge device scenarios
6. **Hyperparameter Tuning**: Automated DP epsilon selection, HE parameter optimization

## References

- **Flower Framework**: https://flower.ai
- **TenSEAL (HE)**: https://github.com/OpenMined/TenSEAL
- **ZKP Implementation**: Based on Pedersen commitments
- **Benchmarking**: Custom framework tracking time, memory, communication

## Support

For issues or questions:
1. Check container logs: `docker compose logs <service>`
2. Review benchmarks in `./results/<mode>/`
3. Test modes individually before full comparison
4. Start with small configurations (2 clients, 3 rounds)
5. Use Docker mode for most reliable results

---

**Last Updated:** January 2025  
**Status:** Baseline ✅ | HE ✅ | ZKP ✅ | DP ✅ (all modes working in Docker)
