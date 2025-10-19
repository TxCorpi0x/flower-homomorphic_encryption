# Differential Privacy (DP) Mode Guide

## Overview

This project now supports **Differential Privacy** as a fourth privacy-preserving approach for federated learning, alongside Baseline, HE, and ZKP.

### Four Privacy-Preserving Modes:

1. **Baseline**: Standard FL without privacy protection (fast, but no privacy guarantees)
2. **HE (Homomorphic Encryption)**: Full parameter encryption (strong privacy, high overhead)
3. **ZKP (Zero-Knowledge Proofs)**: Parameter commitments (moderate privacy, moderate overhead)
4. **DP (Differential Privacy)**: Statistical noise addition (tunable privacy, low overhead)

---

## What is Differential Privacy?

Differential Privacy provides **mathematical guarantees** that individual data points cannot be distinguished in the output, even with auxiliary information.

### Key Concept:
DP adds **calibrated noise** to model parameters/gradients such that:
- The presence or absence of any single training example has minimal impact
- Privacy is quantified by parameters **ε (epsilon)** and **δ (delta)**

---

## DP Mechanism in This Project

### Implementation:

1. **Gradient Clipping**: Bound sensitivity by clipping gradients to max L2 norm
2. **Noise Addition**: Add Gaussian/Laplace noise scaled to privacy budget
3. **Privacy Accounting**: Track cumulative privacy loss across rounds

### Algorithm:

```python
# For each client in each round:
1. Train local model → get parameter updates
2. Clip gradients: clip_norm(updates, C)  # C = max_grad_norm
3. Add noise: updates + N(0, σ²·C²)      # σ = noise_multiplier
4. Send noisy updates to server
```

### Privacy Parameters:

| Parameter | Symbol | Meaning | Typical Values |
|-----------|--------|---------|----------------|
| **Epsilon** | ε | Privacy budget (smaller = more private) | 0.1 - 10.0 |
| **Delta** | δ | Failure probability | 1e-5 - 1e-7 |
| **Max Grad Norm** | C | Clipping threshold | 0.1 - 2.0 |
| **Noise Multiplier** | σ | Scale of added noise | Auto-computed from ε, δ |

---

## Privacy-Utility Tradeoff

### Epsilon (ε) Interpretation:

| Epsilon | Privacy Level | Accuracy Impact | Use Case |
|---------|---------------|-----------------|----------|
| **ε < 0.5** | Very Strong | Significant loss (5-15%) | Medical records, financial data |
| **0.5 ≤ ε < 1.0** | Strong | Moderate loss (2-8%) | Sensitive personal data |
| **1.0 ≤ ε < 3.0** | Moderate | Minimal loss (1-5%) | General purpose protection |
| **ε ≥ 3.0** | Weak | Negligible loss (<2%) | Compliance requirements only |

### Delta (δ) Interpretation:

- **δ = 1e-5**: For datasets with ~100,000 samples
- **δ = 1e-6**: For datasets with ~1,000,000 samples
- **Rule of thumb**: δ < 1/n where n = dataset size

---

## Setup Instructions

### Step 1: Create DP Parameters

```bash
# Default configuration (ε=1.0, δ=1e-5)
python create_dp_params.py

# Custom privacy budget
python create_dp_params.py --epsilon 0.5 --delta 1e-5 --max_grad_norm 1.0

# Very strong privacy
python create_dp_params.py --epsilon 0.1 --delta 1e-6 --max_grad_norm 0.5

# Weak privacy (compliance only)
python create_dp_params.py --epsilon 5.0 --delta 1e-5 --max_grad_norm 2.0
```

This creates `dp_params.pkl` with your DP configuration.

### Step 2: Run DP Mode

**Simulation Mode:**
```bash
python simulation.py simulation \
    --dp \
    --dp_params dp_params.pkl \
    --benchmark \
    --rounds 5 \
    --number_clients 4 \
    --max_epochs 1 \
    --batch_size 32 \
    --device cpu \
    --save_results ./results/dp_test/ \
    --model_save ./results/dp_test/model.pt
```

**Federated Mode (Server):**
```bash
python main_server.py server \
    --dp \
    --rounds 5 \
    --benchmark \
    --model_save ./results/dp_server/model.pt
```

**Federated Mode (Clients):**
```bash
# Client 0
python main_client.py client \
    --dp \
    --dp_params dp_params.pkl \
    --id_client 0 \
    --max_epochs 1 \
    --save_results ./results/dp_client0/

# Client 1
python main_client.py client \
    --dp \
    --dp_params dp_params.pkl \
    --id_client 1 \
    --max_epochs 1 \
    --save_results ./results/dp_client1/
```

### Step 3: Compare All Modes

```bash
python compare_methods_simple.py \
    --modes baseline,he,zkp,dp \
    --rounds 2 \
    --number_clients 2 \
    --max_epochs 1
```

---

## Performance Characteristics

### Expected Performance (2 clients, 1 epoch, CIFAR-10):

| Metric | Baseline | HE | ZKP | DP |
|--------|----------|----|----|-----|
| **Training Time** | ~10s | ~10s | ~10s | ~10s |
| **Crypto Overhead** | 0s | ~0.6s | ~70s | ~0.01s |
| **Communication** | 0.47MB | 0.47MB | 0.47MB | 0.47MB |
| **Accuracy Loss** | 0% | 0-5% | 0-2% | 1-10% (depends on ε) |
| **Privacy Level** | None | Very Strong | Strong | Tunable |

### Advantages of DP:

✅ **Low Computational Overhead**: Just noise addition (~0.01s)
✅ **No Crypto Infrastructure**: No keys, contexts, or commitments
✅ **Tunable Privacy**: Adjust ε for privacy-utility tradeoff
✅ **Mathematical Guarantees**: Formal privacy proof
✅ **Composability**: Track privacy across multiple releases
✅ **Industry Standard**: Used by Google, Apple, Microsoft

### Disadvantages of DP:

❌ **Accuracy Loss**: Noise impacts model quality (especially low ε)
❌ **Privacy Budget Depletion**: ε accumulates across queries
❌ **Hyperparameter Sensitivity**: Requires careful tuning
❌ **No Server-Side Privacy**: Server sees noisy updates (not encrypted)

---

## Comparison: HE vs ZKP vs DP

### Privacy Mechanism:

| Mode | Privacy Method | Server Sees | Privacy Guarantee |
|------|----------------|-------------|-------------------|
| **HE** | Encryption | Encrypted parameters | Computational (IND-CPA) |
| **ZKP** | Commitments | Commitments + proofs | Computational (hiding) |
| **DP** | Statistical noise | Noisy parameters | Information-theoretic (ε-DP) |

### Use Case Recommendations:

| Scenario | Recommended Mode | Reason |
|----------|------------------|--------|
| **Highest Privacy** | HE | Full encryption, no data leakage |
| **Balance Privacy/Speed** | ZKP or DP | Moderate overhead, good privacy |
| **Tunnable Privacy** | DP | Adjust ε for specific requirements |
| **Low Compute Resources** | DP | Minimal overhead |
| **Regulatory Compliance** | DP | Formal guarantees, industry-accepted |
| **Untrusted Server** | HE | Server never sees plaintext |
| **Trusted but Curious Server** | DP or ZKP | Prevent inference attacks |
| **Maximum Speed** | Baseline | No privacy protection |

---

## Benchmarking DP Mode

The benchmark system tracks DP-specific metrics:

### DP Metrics Collected:

```json
{
  "timing": {
    "dp_noise_addition": {
      "mean": 0.015,
      "total": 0.030
    }
  },
  "dp_stats": {
    "clipped": true,
    "original_norm": 2.54,
    "epsilon": 1.0,
    "delta": 1e-5,
    "noise_multiplier": 1.2247
  }
}
```

### Interpreting Results:

- **clipped**: Whether gradients were clipped (true means privacy is enforced)
- **original_norm**: L2 norm before clipping (helps tune max_grad_norm)
- **epsilon**: Privacy budget spent this round
- **noise_multiplier**: Actual noise scale used

---

## Tuning DP Parameters

### General Guidelines:

1. **Start with ε=1.0**: Good balance for most applications
2. **Adjust max_grad_norm**: 
   - Monitor `original_norm` in benchmarks
   - Set `max_grad_norm` slightly above typical values
   - Too low = excessive clipping = poor accuracy
   - Too high = insufficient clipping = weak privacy

3. **Scale noise with data size**:
   - More data → can afford smaller ε
   - Less data → need larger ε to maintain utility

4. **Multi-round privacy**:
   - Privacy budget accumulates: ε_total ≈ ε_per_round × √rounds
   - Use advanced composition theorems for tight bounds

### Recommended Configurations:

**High Privacy (Medical/Financial):**
```bash
python create_dp_params.py \
    --epsilon 0.5 \
    --delta 1e-6 \
    --max_grad_norm 0.5
```

**Standard Privacy (Personal Data):**
```bash
python create_dp_params.py \
    --epsilon 1.0 \
    --delta 1e-5 \
    --max_grad_norm 1.0
```

**Light Privacy (Compliance):**
```bash
python create_dp_params.py \
    --epsilon 3.0 \
    --delta 1e-5 \
    --max_grad_norm 2.0
```

---

## Advanced: Privacy Accounting

### Composition Theorem:

For T rounds of FL with per-round privacy (ε, δ):

**Basic Composition:**
- ε_total = T × ε
- δ_total = T × δ

**Advanced Composition (Moment Accountant):**
- ε_total ≈ ε × √(2T × log(1/δ))
- More tight, used in production systems

### Example:

```python
# 10 rounds with ε=1.0, δ=1e-5 per round

# Basic composition:
ε_total = 10 × 1.0 = 10.0  # Very weak privacy

# Advanced composition:
ε_total ≈ 1.0 × √(2 × 10 × log(1/1e-5)) ≈ 4.8  # Better
```

**Recommendation**: For long training (many rounds), use lower per-round ε.

---

## Troubleshooting

### Problem: Accuracy Too Low

**Solution 1**: Increase epsilon
```bash
python create_dp_params.py --epsilon 2.0  # Instead of 1.0
```

**Solution 2**: Increase max_grad_norm
```bash
python create_dp_params.py --max_grad_norm 2.0  # Instead of 1.0
```

**Solution 3**: Train for more rounds
- DP noise averages out over iterations
- Longer training partially compensates for noise

### Problem: Privacy Too Weak

**Solution**: Decrease epsilon
```bash
python create_dp_params.py --epsilon 0.5  # Instead of 1.0
```

### Problem: "Parameters not found"

**Solution**: Create DP parameters first
```bash
python create_dp_params.py
```

---

## References

- **Original DP Paper**: Dwork et al., "Calibrating Noise to Sensitivity in Private Data Analysis" (2006)
- **DP-SGD**: Abadi et al., "Deep Learning with Differential Privacy" (2016)
- **Privacy Accounting**: Mironov, "Rényi Differential Privacy" (2017)
- **FL + DP**: McMahan et al., "Learning Differentially Private Recurrent Language Models" (2018)

---

## Next Steps

1. ✅ Create DP parameters: `python create_dp_params.py`
2. ✅ Test DP mode: `python simulation.py simulation --dp --dp_params dp_params.pkl`
3. ✅ Compare modes: `python compare_methods_simple.py --modes baseline,he,zkp,dp`
4. 📊 Analyze results: Check `./results/comparison_all_modes/`
5. 🔧 Tune parameters: Adjust ε based on accuracy/privacy tradeoff
6. 📈 Run experiments: Test with different datasets and configurations

---

**✨ DP mode gives you mathematical privacy guarantees with minimal computational overhead!**
