# Docker Guide: Non-Simulation Federated Comparison

This guide shows how to run the Flower server and clients in Docker containers for non-simulation mode across four privacy modes: Baseline, HE (TenSEAL), ZKP, and DP.

## Prerequisites

- Docker Desktop or Docker Engine 20.10+
- docker compose v2
- Local CIFAR-10 data present under `./data/cifar/` (see `README.md` for dataset setup)

## Build the Image

```bash
docker compose build
```

## One-shot: Run All Modes Sequentially

This will initialize keys/params, then run baseline, HE, ZKP, and DP sequentially. Results are saved under `./results/<mode>`.

```bash
bash scripts/run_docker_compare.sh
```

After containers exit, aggregate the results into a single report and plot (using Docker, no local matplotlib needed):

```bash
docker compose --profile aggregate run --rm aggregate
```

This generates:
- `./results/docker_compare/comparison_report.json` - Detailed metrics
- `./results/docker_compare/comparison.png` - Visual comparison plots

## Run a Specific Mode (manually)

1. Initialize keys/params once:

```bash
docker compose --profile init run --rm init
```

2. Baseline:

```bash
docker compose --profile baseline up --abort-on-container-exit
docker compose --profile baseline down -v
```

3. Homomorphic Encryption (TenSEAL):

```bash
docker compose --profile he up --abort-on-container-exit
docker compose --profile he down -v
```

4. Zero-Knowledge Proofs:

```bash
docker compose --profile zkp up --abort-on-container-exit
docker compose --profile zkp down -v
```

5. Differential Privacy:

```bash
docker compose --profile dp up --abort-on-container-exit
docker compose --profile dp down -v
```

## Notes

- Networking: Clients connect to the server using the service hostname (e.g., `server_he:8082`) via `FL_SERVER_ADDRESS`.
- Healthchecks: Clients wait for the server port to be open via simple TCP checks.
- Keys/Params: `create_keys.py`, `create_zkp_params.py`, and `create_dp_params.py` write artifacts into the project directory mounted into all services.
- Results: Per-client training curves and benchmarks are saved under `./results/<mode>`.
- Aggregation/Plots: You can reuse `compare_methods_simple.py` to generate comparison plots from fresh runs, or adapt a simple aggregator to read `client_*_benchmark.json` files.
	- Included: `scripts/aggregate_results.py` to aggregate Docker-run results and generate `comparison_report.json` and `comparison.png`.

## Troubleshooting

- If ports 8081–8084 are in use, change the published ports in `docker-compose.yml`.
- TenSEAL or Concrete-ML build errors: the provided image installs `build-essential` and `cmake`. Ensure sufficient memory for building wheels.
- Dataset not found: Ensure CIFAR-10 exists under `./data/cifar/` as documented.
