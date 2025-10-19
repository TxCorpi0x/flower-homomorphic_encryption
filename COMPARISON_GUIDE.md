# Federated Learning Comparison Guide

## Overview

This project now supports comparing three privacy-preserving federated learning approaches:

1. **Baseline**: Standard federated learning (no privacy protection)
2. **Homomorphic Encryption (HE)**: Encrypts model weights using TenSEAL/CKKS
3. **Zero-Knowledge Proofs (ZKP)**: Proves model integrity using Pedersen commitments

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create HE keys (if comparing HE)
python create_keys.py

# Create ZKP parameters (if comparing ZKP)
python create_zkp_params.py
```

### 2. Run Comparison

**Recommended: Baseline + ZKP (Both work reliably)**
```bash
python compare_methods_simple.py --modes baseline,zkp --rounds 2 --number_clients 2
```

**Note about HE Mode:**  
HE mode currently has compatibility issues with Flower's simulation framework due to TenSEAL object serialization constraints. The framework enforces `allow_pickle=False` for security, but TenSEAL encrypted objects cannot be serialized without pickle support.

**Workaround options:**
1. Compare only Baseline + ZKP (recommended, fully working)
2. Run HE separately using traditional client-server mode (no simulation)
3. Wait for framework updates to better support encrypted object serialization

### 3. View Results

After running, find results in `./results/comparison_all_modes/`:
- `comparison.png` - Visual comparison plots
- `comparison_report.json` - Detailed metrics
- `baseline/benchmark.json` - Baseline metrics
- `zkp/benchmark.json` - ZKP metrics

## Comparison Metrics

The comparison provides:

### Performance Metrics
- **Total Training Time**: End-to-end FL execution time
- **Client Fit Time**: Average time per client training round
- **Server Aggregate Time**: Time for parameter aggregation
- **Communication Overhead**: Upload/download data volume

### Privacy Metrics
- **Cryptographic Overhead**: Time spent on encryption/proofs
- **Proof Generation**: Time to create ZKP proofs (ZKP mode)
- **Proof Verification**: Time to verify proofs (ZKP mode)
- **Encryption/Decryption**: Time for HE operations (HE mode)

### Model Quality Metrics  
- **Training Accuracy**: Local model accuracy per client
- **Validation Accuracy**: Model accuracy on validation set
- **Global Model Accuracy**: Aggregated model performance
- **Loss Convergence**: Initial vs final model loss

## Available Comparison Scripts

### 1. `compare_methods_simple.py` (Recommended)

**Best for: Baseline + ZKP comparison**

Uses simulation mode for both, provides full benchmarking.

```bash
python compare_methods_simple.py \
  --modes baseline,zkp \
  --rounds 2 \
  --number_clients 2 \
  --output_dir ./results/comparison
```

**Pros:**
- Fast execution (simulation mode)
- Full benchmarking support
- Reliable results
- Visual comparison plots

**Cons:**
- HE mode not fully supported due to serialization issues

### 2. `compare_methods.py` (Original)

**Best for: Trying all three modes**

Attempts to run all three modes including HE.

```bash
python compare_methods.py \
  --modes baseline,he,zkp \
  --rounds 2 \
  --number_clients 4
```

**Pros:**
- Includes HE mode attempt
- Comprehensive comparison

**Cons:**
- HE may fail with serialization errors
- Results may be incomplete

### 3. Individual Mode Testing

Test each mode separately:

```bash
# Baseline
python simulation.py simulation --benchmark --rounds 2 --number_clients 2

# ZKP
python simulation.py simulation --zkp --zkp_params zkp_params.pkl --benchmark --rounds 2 --number_clients 2

# HE (may have issues in simulation)
python simulation.py simulation --he --path_keys secret.pkl --path_public_key server_key.pkl --benchmark --rounds 2 --number_clients 2
```

## Understanding Results

### Typical Performance Profile

**Baseline** (Fast, no privacy):
- Training: ~20s for 2 rounds
- Crypto overhead: 0s
- Accuracy: ~43%

**ZKP** (Medium overhead, proof-based privacy):
- Training: ~20s + ~140s crypto overhead
- Crypto: Proof generation + verification
- Accuracy: ~42% (similar to baseline)
- Communication: Slightly higher (includes commitments)

**HE** (High overhead, encrypted computation):
- Training: Variable (depends on encryption efficiency)
- Crypto: Encryption + decryption overhead
- Accuracy: Should match baseline
- Communication: Much higher (encrypted parameters larger)

### Performance Trade-offs

| Mode | Privacy Level | Speed | Communication | Accuracy Impact |
|------|--------------|-------|---------------|-----------------|
| Baseline | None | Fast | Low | Baseline |
| ZKP | High (Integrity) | Medium | Medium | Minimal |
| HE | Highest (Confidentiality) | Slow | High | Minimal |

## Troubleshooting

### HE Serialization Error

**Error:** `ValueError: Object arrays cannot be saved when allow_pickle=False`

**Cause:** Flower simulation cannot serialize TenSEAL encrypted objects.

**Solution:** 
- Use baseline + ZKP comparison (both work reliably)
- Run HE in traditional client-server mode (not simulation)
- Wait for framework updates

### ZKP Parameter Error

**Error:** `FileNotFoundError: zkp_params.pkl`

**Solution:**
```bash
python create_zkp_params.py
```

### HE Keys Missing

**Error:** `FileNotFoundError: secret.pkl`

**Solution:**
```bash
python create_keys.py
```

### Ray Memory Issues

**Error:** `OutOfMemoryError` or actor crashes

**Solution:**
- Reduce `--number_clients`
- Reduce `--batch_size`
- Use fewer concurrent actors

## Configuration Options

```bash
python compare_methods_simple.py \
  --modes baseline,zkp \          # Modes to compare
  --rounds 5 \                     # FL rounds
  --number_clients 4 \             # Number of clients
  --max_epochs 1 \                 # Local epochs per round
  --batch_size 64 \                # Training batch size
  --device cpu \                   # Device (cpu/cuda/mps)
  --output_dir ./results/my_test   # Output directory
```

## Best Practices

1. **Start Small**: Test with 2 clients, 2 rounds first
2. **Use Baseline + ZKP**: Most reliable comparison currently
3. **Monitor Resources**: Ray simulation can be memory-intensive
4. **Check Logs**: Look at `stdout.log` and `stderr.log` in result directories
5. **Validate Benchmarks**: Ensure `benchmark.json` has non-zero values

## Implementation Status

- ✅ **Baseline Mode**: Fully working with benchmarking
- ✅ **ZKP Mode**: Fully working with benchmarking  
- ⚠️ **HE Mode**: Works standalone, serialization issues in simulation
- ✅ **Comparison Framework**: Working for baseline + ZKP
- ✅ **Visualization**: 6-plot comparison dashboard
- ✅ **Model Quality Metrics**: Train/val/test accuracy and loss tracking

## Future Improvements

Potential enhancements:

1. **Fix HE Serialization**: Modify HE to serialize to raw bytes before Flower serialization
2. **Differential Privacy**: Add DP noise to gradients
3. **Secure Aggregation**: Implement secure multi-party computation
4. **More Datasets**: Support MNIST, Fashion-MNIST, etc.
5. **GPU Support**: Enable GPU acceleration for faster training

## References

- **Flower Framework**: https://flower.ai
- **TenSEAL (HE)**: https://github.com/OpenMined/TenSEAL
- **ZKP Implementation**: Based on Pedersen commitments
- **Benchmarking**: Custom framework tracking time, memory, communication

## Support

For issues or questions:
1. Check logs in result directories
2. Review error messages in `stderr.log`
3. Test modes individually before comparison
4. Start with small configurations (2 clients, 2 rounds)
5. Use baseline + ZKP for reliable results

---

**Last Updated:** October 19, 2025  
**Status:** Baseline ✅ | ZKP ✅ | HE ⚠️ (serialization issue)
