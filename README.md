# Privacy-Preserving Federated Learning Framework# Privacy-Preserving Federated Learning Framework



[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[![Flower 1.8.0](https://img.shields.io/badge/flower-1.8.0-green.svg)](https://flower.ai)[![Flower 1.8.0](https://img.shields.io/badge/flower-1.8.0-green.svg)](https://flower.ai)

[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)



A comprehensive federated learning framework implementing **four privacy-preserving approaches** with Docker support, automated benchmarking, and detailed comparison tools.A comprehensive federated learning framework implementing **four privacy-preserving approaches** with Docker support, automated benchmarking, and detailed comparison tools.



## 🎯 Overview## 🚀 Features



This framework enables privacy-preserving federated learning with multiple protection mechanisms:- **Four Privacy Modes**:

  - 🔓 **Baseline**: Standard FL (no privacy)

| Mode | Privacy Mechanism | Overhead | Best For |  - 🔐 **HE (Homomorphic Encryption)**: TenSEAL CKKS encryption with real encrypted transport

|------|------------------|----------|----------|  - ✓ **ZKP (Zero-Knowledge Proofs)**: Pedersen commitment verification

| **Baseline** | None | Minimal | Performance baseline |  - 🎲 **DP (Differential Privacy)**: Gaussian noise with gradient clipping

| **HE** | Homomorphic Encryption (TenSEAL CKKS) | ~3-4s crypto | Maximum privacy, untrusted aggregator |

| **ZKP** | Zero-Knowledge Proofs (Pedersen) | ~93s crypto | Verifiable computation |- **Docker-First Workflow**: Complete containerized execution

| **DP** | Differential Privacy (Gaussian) | <0.1s | Regulatory compliance, tunable |- **Comprehensive Benchmarking**: Timing, memory, communication, accuracy

- **Automated Comparison**: Run all modes and generate visualizations

## 🚀 Quick Start (Docker - Recommended)- **Production-Ready**: Real non-simulation mode with gRPC transport

## Configure an environment

### 1. Prerequisites1. Creating an environment

    ```

```bash    conda create -n fl_env python=3.10 anaconda

# Ensure Docker and Docker Compose are installed    ```

docker --version2. Add the necessary libraries with pip

docker compose version   - If you want to install a specific version of Flower :

```      - You can use this command (example with version 1.4.0) : : `pip install flower==1.4.0` 

      - You have to change `parameters.py` and `__init__.py` in flower library from this folder : flower/src/py/flwr/common (see modification in this github : https://github.com/data-science-lover/flower.git)

### 2. Run All Modes

   ```

```bash   pip install -r requirements.txt

# Initialize cryptographic parameters   pip install pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

docker compose --profile init run --rm init   git clone https://github.com/data-science-lover/flower.git

   cd flower

# Run baseline mode   pip install .

docker compose --profile baseline up --abort-on-container-exit   ```

3. Activate the environment

# Run HE mode   ```

docker compose --profile he up --abort-on-container-exit   conda activate fl_env

   ```

# Run ZKP mode4. Add a specific dataset

docker compose --profile zkp up --abort-on-container-exit   - If folder where put the datasets (for example called `data`) doesn't exist : create this folder and subfolder with the name of the specific dataset (example `cifar`): `./data/cifar`

   - In data, you have all datasets and in a specific dataset you have train and test folder: `./data/cifar/train/` 

# Run DP mode   - If you have a specific dataset for the validation then addi the path to the folder with this command : `--data_path_val data/cifar/val` 

docker compose --profile dp up --abort-on-container-exit   - To create train and test folder : run in python console the `create_files_train_test` function from `core/common.py`.

   - Open `core/data_setup.py` :

# Generate comparison report     - Add normalize values specified to the dataset in the `NORMALIZE_DICT`

docker compose --profile aggregate run --rm aggregate     - If the dataset comes from the torchvision library, add a condition similar to that of CIFAR in `load_datasets` function

```   - Open `core/common.py` and add condition similar to the other datasets in `classes_string`

   - When you want to run an algorithme, add the validation path if you have a specific folder for this (None by default): --data_path_val ./data/histo/val

Results are saved in:## Classic classifier (centralized training)

### Launch the training

- `./results/baseline/` - Baseline benchmarks```

- `./results/he_tenseal/` - HE benchmarkspython classic.py run --data_path data/ --dataset cifar --yaml_path ./results/classic/results.yml --seed 42 --num_workers -1 --max_epochs 5 --batch_size 32 --length 32 --split 10 --device mps --save_results results/classic/ --matrix_path confusion_matrix.png --roc_path roc.png --model_save cifar.pt

- `./results/zkp/` - ZKP benchmarks```

- `./results/dp/` - DP benchmarks## Federated Learning

- `./results/docker_compare/` - Aggregated report and plots### A. Train on local machine (Launch the simulation)

```

### 3. View Resultspython simulation.py simulation --data_path data/ --dataset cifar --yaml_path ./results/FL/results.yml --seed 42 --num_workers -1 --max_epochs 5 --batch_size 32 --length 32 --split 10 --device mps --number_clients 10 --save_results results/FL/ --matrix_path confusion_matrix.png --roc_path roc.png --model_save cifar_fl.pt --min_fit_clients 10 --min_avail_clients 10 --min_eval_clients 10 --rounds 2 --frac_fit 1.0 --frac_eval 0.5

```

```bash

# Check the comparison report### B. Train without simulation

cat ./results/docker_compare/comparison_report.json1) Run the central server



# View the plot   Open a terminal windows for the central server and run the client script client.py

open ./results/docker_compare/comparison.png  # macOS   ```

xdg-open ./results/docker_compare/comparison.png  # Linux   python main_server.py server --data_path data/ --dataset cifar --seed 42 --num_workers 0 --max_epochs 5 --batch_size 32 --length 32 --split 10 --device mps --number_clients 3 --min_fit_clients 2 --min_avail_clients 2 --min_eval_clients 2 --rounds 2 --frac_fit 1.0 --frac_eval 0.5

```   ```

2) Run each client

## 💻 Local Installation (Alternative)

   Open a terminal windows for each client and run the client script client.py (change value for the client ID) 

### Setup Environment   ```

   python main_client.py client --data_path data/ --dataset cifar --seed 42 --num_workers 0 --max_epochs 5 --batch_size 32 --length 32 --split 10 --device mps --number_clients 3 --save_results results/FL/ --matrix_path confusion_matrix2.png --roc_path roc2.png --id_client 0

```bash   ```

# Create conda environment

conda create -n fl_env python=3.10## To use Zero-Knowledge Proofs (ZKP)

conda activate fl_env

ZKP provides an alternative to HE for privacy-preserving FL with different performance characteristics:

# Install dependencies

pip install -r requirements.txt1. **Generate ZKP parameters** (one-time setup):

```   ```bash

   python create_zkp_params.py

### Run Locally   ```

   This creates `zkp_params.pkl` with cryptographic parameters (Pedersen commitment scheme).

```bash

# Generate keys/parameters2. **Run with ZKP**:

python create_keys.py          # For HE   ```bash

python create_zkp_params.py    # For ZKP   # Simulation

python create_dp_params.py     # For DP   python simulation.py simulation --zkp --zkp_params zkp_params.pkl \

     --data_path data/ --dataset cifar --number_clients 4 --rounds 2

# Run simulation (single process)

python simulation.py simulation --rounds 3 --number_clients 2 --benchmark   # Server (for client-server mode)

   python main_server.py server --zkp --zkp_params zkp_params.pkl \

# Run with privacy mode     --data_path data/ --dataset cifar --number_clients 3 --rounds 2

python simulation.py simulation --he --rounds 3 --benchmark

python simulation.py simulation --zkp --zkp_params zkp_params.pkl --rounds 3 --benchmark   # Client (for client-server mode)

python simulation.py simulation --dp --dp_params dp_params.pkl --rounds 3 --benchmark   python main_client.py client --zkp --zkp_params zkp_params.pkl \

```     --data_path data/ --dataset cifar --id_client 0

   ```

## 📊 Performance Comparison

**Note**: Unlike HE, ZKP uses the same parameters for both client and server.

Based on 3 rounds, 2 clients, CIFAR-10, CPU (from Docker runs):

## Performance Comparison: Baseline vs HE vs ZKP

| Mode | Total Time | Fit Time (avg) | Crypto Overhead | Accuracy | Communication |

|------|-----------|---------------|-----------------|----------|---------------|Compare all three methods automatically:

| Baseline | ~116s | ~39s | 0s | ~38% | 1.4 MB |

| HE | ~120s | ~40s | ~3.7s | ~45% | 328 MB |```bash

| ZKP | ~128s | ~43s | ~93s | ~39% | 1.4 MB |# Quick comparison

| DP | ~127s | ~42s | <0.1s | ~42% | 1.4 MB |python compare_methods.py \

  --modes baseline,he,zkp \

**Key Insights:**  --number_clients 4 \

  --rounds 2 \

- **HE**: Best accuracy with real encrypted transport, high communication overhead  --output_dir ./results/comparison

- **ZKP**: High crypto overhead for proof generation (~31s per round for fc3.weight)

- **DP**: Minimal overhead, tunable privacy-accuracy tradeoff# This will:

- **Baseline**: Reference performance, no privacy# - Run experiments with all three modes

# - Generate comparison plots

## 🏗️ Architecture# - Save detailed benchmarks

# - Print performance summary

### Model```



- **LeNet-5 style CNN** (~62K parameters)### Enable Benchmarking

- 2 Conv layers + 3 FC layers

- Trained on CIFAR-10 (10 classes, 32×32 images)Add `--benchmark` to any experiment to track performance metrics:



### Framework```bash

python simulation.py simulation --benchmark --zkp \

- **Flower 1.8.0**: Federated learning orchestration  --data_path data/ --dataset cifar --number_clients 4 --rounds 2

- **TenSEAL**: CKKS homomorphic encryption```

- **PyTorch**: Model training

- **Docker**: Containerized executionMetrics tracked:

- Timing (training, aggregation, crypto operations)

### Project Structure- Memory usage (peak client/server)

- Communication overhead (upload/download sizes)

```text- Cryptographic overhead (encryption, proofs)

.

├── docker-compose.yml          # Docker orchestration**For detailed documentation, see [ZKP_GUIDE.md](ZKP_GUIDE.md)**

├── Dockerfile                  # Container image

├── requirements.txt            # Python dependencies## To use the homomorphic encryption

│If you have the TenSEAL private/public key combination, you can use homomorphic encryption by adding the command `--he` at client and server sites. 

├── core/                       # Core implementations

│   ├── model_builder.py        # CNN modelThe private key must be on the client side and the public key on the server side, but the private key must be the same for all clients because all weights must be encrypted with the same private key. 

│   ├── security.py             # HE + DP

│   ├── zkp.py                  # ZKP primitivesOtherwise, you can create the combined private/public keys on a common entity (not the aggregation server) by running the create_keys.py script: `create_keys.py` script : 

│   ├── benchmark.py            # Metrics tracking```

│   ├── common.py               # Shared utilitiespython create_keys.py

│   ├── data_setup.py           # Dataset loading```

│   └── engine.py               # Training logic

│Warning :

├── client.py                   # FL client logic- you have to define the path for the crypted results : `--path_crypted server.pkl` (The crypted (and not crypted) weights are saved by default in "server.pkl" file)

├── server.py                   # FL server strategy- server side : 

├── main_client.py              # Client entry point  -  you must have the public key (by default in "server_key.pkl" file) : `--path_public_key server_key.pkl`

├── main_server.py              # Server entry point- client side : 

│  - You must have the combo private/public keys (by default in "secret.pkl" file): `--path_keys secret.pkl`

├── create_keys.py              # Generate HE keys

├── create_zkp_params.py        # Generate ZKP parameters## References

├── create_dp_params.py         # Generate DP parameters

│The federated learning framework used is https://github.com/adap/flower and the HE library used is https://github.com/OpenMined/TenSEAL, please refer to their documentation for more information.

├── scripts/
│   ├── aggregate_results.py    # Aggregate benchmarks
│   └── run_docker_compare.sh   # Run all modes
│
└── results/                    # Output directory
    ├── baseline/
    ├── he_tenseal/
    ├── zkp/
    ├── dp/
    └── docker_compare/
```

## 🔐 Privacy Mechanisms Explained

### Homomorphic Encryption (HE)

**What it does:**

- Encrypts model parameters with CKKS scheme
- Server aggregates encrypted parameters without decryption
- Provides cryptographic confidentiality

**Use when:**

- Server cannot be trusted with model parameters
- Maximum privacy is required
- Communication bandwidth is available

**Configuration:**

```python
# In core/security.py context()
poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale = 2**40
```

### Zero-Knowledge Proofs (ZKP)

**What it does:**

- Creates Pedersen commitments for model parameters
- Proves parameter integrity without revealing values
- Enables verification without decryption

**Use when:**

- Need verifiable computation
- Audit trails are required
- Parameter integrity is critical

**Configuration:**

```python
# In create_zkp_params.py
bit_length = 2048  # Security parameter
```

### Differential Privacy (DP)

**What it does:**

- Adds calibrated Gaussian noise to gradients
- Clips gradients to bound sensitivity
- Provides formal privacy guarantees (ε-DP)

**Use when:**

- Regulatory compliance (GDPR, HIPAA)
- Tunable privacy-utility tradeoff needed
- Minimal overhead required

**Configuration:**

```python
# In create_dp_params.py
epsilon = 1.0      # Privacy budget (lower = more private)
delta = 1e-5       # Failure probability
max_grad_norm = 1.0  # Clipping threshold
```

## 🐳 Docker Profiles

The `docker-compose.yml` defines several profiles:

| Profile | Purpose | Services |
|---------|---------|----------|
| `init` | Generate keys/params | init container |
| `baseline` | Standard FL | server_baseline + 2 clients |
| `he` | Homomorphic encryption | server_he + 2 clients |
| `zkp` | Zero-knowledge proofs | server_zkp + 2 clients |
| `dp` | Differential privacy | server_dp + 2 clients |
| `aggregate` | Generate comparison report | aggregate container |

Each mode runs on a separate port:

- Baseline: `8081`
- HE: `8082`
- ZKP: `8083`
- DP: `8084`

## 📈 Benchmarking

The framework tracks comprehensive metrics:

### Timing Metrics

- `client_fit`: Local training time per round
- `server_aggregate`: Parameter aggregation time
- `encryption/decryption`: HE operation time
- `proof_generation/verification`: ZKP operation time
- `dp_noise_addition`: DP operation time

### Resource Metrics

- `client_peak`: Peak client memory (MB)
- `server_peak`: Peak server memory (MB)
- `upload/download`: Communication volume (MB)

### Model Quality Metrics

- `train_loss/accuracy`: Local training metrics
- `val_loss/accuracy`: Validation metrics
- `global_val_loss/accuracy`: Aggregated model metrics

All metrics are saved to `benchmark.json` files for each run.

## 🔧 Advanced Usage

### Non-Simulation Mode (Client-Server)

Run server and clients as separate processes:

```bash
# Terminal 1: Start server
python main_server.py server \
  --rounds 3 \
  --number_clients 2 \
  --benchmark

# Terminal 2: Start client 0
python main_client.py client \
  --id_client 0 \
  --number_clients 2 \
  --benchmark

# Terminal 3: Start client 1
python main_client.py client \
  --id_client 1 \
  --number_clients 2 \
  --benchmark
```

### Custom Dataset

1. Prepare data structure:

```text
./data/your_dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

2. Update `core/data_setup.py`:

```python
# Add normalization values
NORMALIZE_DICT["your_dataset"] = ([mean_r, mean_g, mean_b], [std_r, std_g, std_b])

# Add dataset loading logic
```

3. Run with your dataset:

```bash
python simulation.py simulation \
  --dataset your_dataset \
  --data_path ./data/your_dataset/
```

### Tuning DP Parameters

```bash
# Stricter privacy (lower ε)
python create_dp_params.py --epsilon 0.5 --delta 1e-5

# More utility (higher ε)
python create_dp_params.py --epsilon 5.0 --delta 1e-5

# Tighter gradient clipping
python create_dp_params.py --max_grad_norm 0.5
```

## 🔍 Troubleshooting

### Docker Issues

**Problem:** TenSEAL build fails on Apple Silicon

**Solution:** The Dockerfile specifies `platform: linux/amd64` to ensure x86_64 binaries.

**Problem:** Port already in use

**Solution:** Change ports in `docker-compose.yml` or stop conflicting services.

### Key/Parameter Issues

**Problem:** `FileNotFoundError: secret.pkl`

**Solution:** Run `docker compose --profile init run --rm init` or `python create_keys.py`

**Problem:** HE "scale out of bounds" error

**Solution:** The current implementation uses normalized weights to avoid scale overflow. If you modify the aggregation, ensure scales remain bounded.

### Performance Issues

**Problem:** Training very slow

**Solution:**

- Reduce `--number_clients`
- Reduce `--batch_size`
- Use GPU with `--device cuda` (requires CUDA-capable Docker)
- Reduce dataset size

## 📚 Documentation

- [Docker Guide](DOCKER_GUIDE.md) - Detailed Docker setup
- [DP Guide](DP_GUIDE.md) - Differential Privacy details
- [ZKP Guide](ZKP_GUIDE.md) - Zero-Knowledge Proof details
- [Comparison Guide](COMPARISON_GUIDE.md) - Performance analysis

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional privacy mechanisms (secure aggregation, TEE)
- More datasets (ImageNet, medical imaging)
- GPU acceleration
- Byzantine-robust aggregation
- Model compression techniques
- Cross-device FL scenarios

## 📄 License

This project is open source. Please add appropriate license before public release.

## 🙏 Acknowledgments

- **Flower Framework**: [https://flower.ai](https://flower.ai)
- **TenSEAL**: [https://github.com/OpenMined/TenSEAL](https://github.com/OpenMined/TenSEAL)
- Presented at [Flower Summit 2023](https://youtu.be/pAvex7tpq2w?si=_sOmVMjiyA3cI0E5)

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for privacy-preserving machine learning**
