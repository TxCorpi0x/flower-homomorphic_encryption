# Zero-Knowledge Proof and Performance Comparison Guide

This guide covers the new Zero-Knowledge Proof (ZKP) functionality and how to compare performance between Baseline, Homomorphic Encryption (HE), and ZKP modes.

## Table of Contents
1. [Overview](#overview)
2. [Zero-Knowledge Proof Support](#zero-knowledge-proof-support)
3. [Performance Benchmarking](#performance-benchmarking)
4. [Comparing Methods](#comparing-methods)
5. [Examples](#examples)

## Overview

This project now supports three privacy-preserving modes for federated learning:

1. **Baseline**: Standard FL without cryptographic protection (fast, but no privacy guarantees)
2. **Homomorphic Encryption (HE)**: Uses TenSEAL/CKKS to encrypt model parameters
3. **Zero-Knowledge Proofs (ZKP)**: Uses Pedersen commitments for parameter verification

Each mode has different trade-offs in terms of:
- **Security**: Level of privacy protection
- **Performance**: Computational and communication overhead
- **Complexity**: Setup and implementation complexity

## Zero-Knowledge Proof Support

### What is ZKP in this context?

The ZKP implementation uses **Pedersen commitments** to allow clients to prove they correctly updated their model parameters during training without revealing the actual parameter values. This provides:

- **Privacy**: Server cannot see actual parameter values
- **Verifiability**: Server can verify parameters were computed correctly
- **Efficiency**: More computationally efficient than full homomorphic encryption for certain operations

### How it works

1. **Client side**:
   - Trains model locally
   - Creates Pedersen commitments for each parameter: `C = g^m * h^r mod p`
   - Sends commitments (not raw parameters) to server
   - Can provide zero-knowledge proofs of correct computation

2. **Server side**:
   - Receives commitments from clients
   - Performs aggregation on commitments
   - Verifies proofs without learning parameter values
   - Returns aggregated commitments to clients

3. **Protected layers**:
   - By default, only `fc3.weight` (final layer) is protected with ZKP
   - Other layers sent in clear for efficiency
   - Configurable in `core/zkp.py::zkp_commit_model()`

### Setup ZKP Parameters

Before using ZKP mode, generate the cryptographic parameters:

```bash
python create_zkp_params.py
```

Options:
```bash
python create_zkp_params.py --params_path zkp_params.pkl --bit_length 2048
```

- `--params_path`: Where to save ZKP parameters (default: `zkp_params.pkl`)
- `--bit_length`: Security parameter in bits (default: 2048)

This creates a file with:
- Large prime `p` (modulus)
- Prime `q` (group order)
- Generators `g` and `h` for Pedersen commitments

⚠️ **Important**: Unlike HE, the same ZKP parameters can be used by both clients and server.

### Running with ZKP

#### Simulation mode

```bash
python simulation.py simulation --zkp --zkp_params zkp_params.pkl \
  --data_path ./data/ --dataset cifar --number_clients 4 \
  --rounds 2 --max_epochs 1 --batch_size 32 --device cpu
```

#### Client-server mode

1. Start server:
```bash
python main_server.py server --zkp --zkp_params zkp_params.pkl \
  --data_path ./data/ --dataset cifar --number_clients 3 \
  --rounds 2 --device cpu
```

2. Start clients:
```bash
python main_client.py client --zkp --zkp_params zkp_params.pkl \
  --data_path ./data/ --dataset cifar --id_client 0 --device cpu
```

### ZKP vs HE Comparison

| Feature | HE (TenSEAL) | ZKP (Pedersen) |
|---------|--------------|----------------|
| **Privacy** | Full encryption of parameters | Commitments hide parameter values |
| **Computation** | Heavy (multiplications in encrypted domain) | Light (modular exponentiation) |
| **Communication** | Large (serialized ciphertexts) | Moderate (commitments) |
| **Aggregation** | Homomorphic operations | Commitment arithmetic |
| **Verification** | No inherent verification | Can verify correctness |
| **Key Management** | Shared secret key required | Same public params for all |

## Performance Benchmarking

### Enable Benchmarking

Add `--benchmark` flag to any experiment:

```bash
python simulation.py simulation --benchmark --zkp --zkp_params zkp_params.pkl \
  --data_path ./data/ --dataset cifar --number_clients 4 --rounds 2
```

This tracks:
- **Timing**: Client fit/eval, server aggregation, crypto operations
- **Memory**: Peak memory usage (client and server)
- **Communication**: Upload/download sizes
- **Cryptographic overhead**: Encryption, decryption, proof generation/verification

### Benchmark Output

Results are saved to `benchmark.json` with structure:

```json
{
  "mode": "zkp",
  "num_clients": 4,
  "rounds": 2,
  "timing": {
    "client_fit": {"mean": 2.5, "std": 0.1, "total": 20.0},
    "encryption": {"mean": 0.3, "std": 0.05, "total": 2.4},
    ...
  },
  "memory_mb": {
    "client_peak": {"mean": 512, "max": 550},
    ...
  },
  "communication_bytes": {
    "upload": {"total": 5242880},
    ...
  }
}
```

## Comparing Methods

### Quick Comparison Script

The `compare_methods.py` script runs all three modes automatically and generates comparison plots:

```bash
python compare_methods.py \
  --modes baseline,he,zkp \
  --number_clients 4 \
  --rounds 2 \
  --max_epochs 1 \
  --dataset cifar \
  --output_dir ./results/comparison
```

Options:
- `--modes`: Comma-separated list of modes (default: `baseline,he,zkp`)
- `--output_dir`: Where to save results (default: `./results/comparison`)
- `--number_clients`: Number of FL clients
- `--rounds`: Number of FL rounds
- `--max_epochs`: Local epochs per round
- `--dataset`: Dataset to use
- `--device`: Device (cpu, cuda, mps)

### Prerequisites

Before running comparisons:

1. **For HE mode**:
   ```bash
   python create_keys.py
   ```
   Creates `secret.pkl` and `server_key.pkl`

2. **For ZKP mode**:
   ```bash
   python create_zkp_params.py
   ```
   Creates `zkp_params.pkl`

3. **Data**: Ensure dataset is available at `--data_path`

### Output

The comparison script generates:

1. **Individual run results**: 
   - `{output_dir}/baseline/`
   - `{output_dir}/he/`
   - `{output_dir}/zkp/`

2. **Comparison plot**: `{output_dir}/comparison.png`
   - Total training time
   - Client fit time per round
   - Communication overhead
   - Cryptographic overhead

3. **Summary report**: `{output_dir}/comparison_report.json`
   - All benchmark data
   - Success status
   - Timing breakdown

4. **Console summary**:
   ```
   ============================================================
   PERFORMANCE SUMMARY
   ============================================================
   
   Mode         Total Time      Avg Fit         Comm (MB)       Crypto         
   ------------------------------------------------------------------------
   BASELINE     45.23           1.131           23.50           0.00           
   HE           156.78          3.921           45.20           67.30          
   ZKP          78.45           1.963           28.90           12.50          
   ```

## Examples

### Example 1: Quick ZKP Test

```bash
# Generate ZKP parameters
python create_zkp_params.py

# Run simulation with ZKP
python simulation.py simulation --zkp --zkp_params zkp_params.pkl \
  --data_path ./data/ --dataset cifar \
  --number_clients 2 --rounds 1 --max_epochs 1 \
  --batch_size 32 --device cpu --benchmark \
  --save_results ./results/zkp_test/
```

### Example 2: Full Comparison

```bash
# Setup (if not already done)
python create_keys.py           # For HE
python create_zkp_params.py     # For ZKP

# Run comprehensive comparison
python compare_methods.py \
  --modes baseline,he,zkp \
  --number_clients 4 \
  --rounds 3 \
  --max_epochs 2 \
  --dataset cifar \
  --batch_size 64 \
  --device mps \
  --output_dir ./results/full_comparison
```

### Example 3: HE vs ZKP Only

```bash
python compare_methods.py \
  --modes he,zkp \
  --number_clients 3 \
  --rounds 2 \
  --output_dir ./results/crypto_comparison
```

### Example 4: Benchmarking Single Mode

```bash
# Baseline with benchmarking
python simulation.py simulation --benchmark \
  --data_path ./data/ --dataset cifar \
  --number_clients 5 --rounds 3 \
  --save_results ./results/baseline_bench/

# Check results
cat ./results/baseline_bench/benchmark.json
```

## Understanding the Results

### Performance Expectations

**Baseline** (fastest, no privacy):
- Training time: ~1x (reference)
- Communication: Smallest
- Memory: Lowest

**ZKP** (medium overhead, verification):
- Training time: ~1.5-2x baseline
- Communication: ~1.2-1.5x baseline
- Memory: Moderate
- Extra: Proof generation/verification time

**HE** (highest overhead, full encryption):
- Training time: ~3-5x baseline
- Communication: ~2-3x baseline (serialized ciphertexts)
- Memory: Highest
- Extra: Encryption/decryption time

### When to Use Each Mode

| Mode | Use When... |
|------|-------------|
| **Baseline** | - No privacy concerns<br>- Need maximum performance<br>- Trusted environment |
| **ZKP** | - Need parameter verification<br>- Moderate privacy requirements<br>- Constrained compute resources<br>- Want balance of security/performance |
| **HE** | - Need full parameter encryption<br>- Highest privacy requirements<br>- Can tolerate performance overhead<br>- Have computational resources |

## Troubleshooting

### ZKP Issues

**Error: `ZKP parameters not found`**
```bash
python create_zkp_params.py
```

**ZKP verification failed**
- Check that all clients use the same `zkp_params.pkl`
- Ensure parameters weren't corrupted during transfer

**Slow ZKP commitment generation**
- Reduce `--bit_length` (trade security for speed)
- Reduce number of protected layers in `core/zkp.py`

### Comparison Issues

**Missing mode in comparison**
```bash
# Check prerequisites
ls secret.pkl server_key.pkl zkp_params.pkl

# Generate missing files
python create_keys.py          # For HE
python create_zkp_params.py    # For ZKP
```

**Out of memory during comparison**
- Reduce `--number_clients`
- Reduce `--rounds` or `--max_epochs`
- Use smaller `--batch_size`
- Run modes sequentially instead of all at once

## API Reference

### Core Modules

- `core/zkp.py`: ZKP primitives (Pedersen commitments, proofs)
- `core/benchmark.py`: Performance tracking utilities
- `compare_methods.py`: Automated comparison script
- `create_zkp_params.py`: ZKP parameter generation

### Key Functions

```python
# ZKP
from core.zkp import create_zkp_context, zkp_commit_model, verify_zkp_layers

context = create_zkp_context(bit_length=2048)
zkp_layers = zkp_commit_model(model.state_dict(), context)
is_valid = verify_zkp_layers(zkp_layers)

# Benchmarking
from core.benchmark import init_benchmark, BenchmarkTimer

metrics = init_benchmark("zkp", num_clients=4, rounds=2)
with BenchmarkTimer(metrics, 'client_fit'):
    # ... training code ...
metrics.save("benchmark.json")
```

## Further Reading

- [Pedersen Commitments](https://en.wikipedia.org/wiki/Commitment_scheme)
- [Zero-Knowledge Proofs in FL](https://arxiv.org/abs/2109.12677)
- [TenSEAL Documentation](https://github.com/OpenMined/TenSEAL)
- [Flower Documentation](https://flower.dev/docs/)
