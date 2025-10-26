"""
Simple comparison script that works around HE simulation issues.

Strategy:
- Baseline: simulation mode with benchmarking ✓
- ZKP: simulation mode with benchmarking ✓
- HE: simulation mode BUT decrypt parameters on server before aggregation ✓

This allows all modes to use the same simulation infrastructure and benchmarking.
"""

import argparse
import subprocess
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import signal
import sys


def run_experiment_distributed(
    mode, base_args, output_dir, result_dir, he_backend, display_mode
):
    """Run experiment in non-simulation mode with separate server/client processes."""

    # Build common arguments (shared between server and client)
    common_args = []

    arg_mapping = {
        "dataset": "--dataset",
        "data_path": "--data_path",
        "max_epochs": "--max_epochs",
        "batch_size": "--batch_size",
        "device": "--device",
    }

    for key, flag in arg_mapping.items():
        if key in base_args and base_args[key] is not None and base_args[key] != "":
            common_args.extend([flag, str(base_args[key])])

    # Server-specific arguments
    server_only_args = []
    server_arg_mapping = {
        "rounds": "--rounds",
        "number_clients": "--number_clients",
    }

    for key, flag in server_arg_mapping.items():
        if key in base_args and base_args[key] is not None and base_args[key] != "":
            server_only_args.extend([flag, str(base_args[key])])

    # Mode-specific flags
    mode_args = []
    if mode == "he":
        mode_args.append("--he")
        if he_backend:
            mode_args.extend(["--he_backend", he_backend])
        mode_args.extend(["--path_keys", "secret.pkl"])
        mode_args.extend(["--path_public_key", "server_key.pkl"])
    elif mode == "zkp":
        mode_args.append("--zkp")
        mode_args.extend(["--zkp_params", "zkp_params.pkl"])
    elif mode == "dp":
        mode_args.append("--dp")
        mode_args.extend(["--dp_params", "dp_params.pkl"])

    # Benchmarking (only if supported by the subcommand)
    benchmark_args = ["--benchmark"]

    # Start server in background
    server_cmd = (
        [sys.executable, "main_server.py", "server"]
        + common_args
        + server_only_args
        + mode_args
        + benchmark_args
    )
    print(f"Starting server: {' '.join(server_cmd)}\n")

    # Use a dedicated port to avoid conflicts and pass it via env
    port_map = {"baseline": 8081, "he": 8082, "zkp": 8083, "dp": 8084}
    base_mode = "he" if mode == "he" else (mode if mode in port_map else "baseline")
    server_addr = f"127.0.0.1:{port_map.get(base_mode, 8081)}"
    env_server = os.environ.copy()
    env_server["FL_SERVER_ADDRESS"] = server_addr

    server_log_path = f"{result_dir}/server.log"
    with open(server_log_path, "w") as server_log:
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            env=env_server,
        )

    # Give server time to start
    time.sleep(5)  # Increased from 3 to 5 seconds

    # Check if server is still running
    if server_proc.poll() is not None:
        print(f"❌ Server failed to start! Check {server_log_path}")
        return {
            "mode": display_mode,
            "success": False,
            "exit_code": server_proc.returncode,
            "result_dir": result_dir,
            "benchmark": None,
        }

    print(f"✓ Server started (PID: {server_proc.pid})")

    # Client-specific benchmark args (parser_ml is available for client subcommand)
    client_benchmark_args = benchmark_args + [
        "--save_results",
        result_dir,
        "--model_save",
        f"{result_dir}/model.pt",
    ]

    # Start clients in parallel (not sequentially)
    num_clients = base_args.get("number_clients", 2)
    client_procs = []
    client_log_paths = []

    for cid in range(num_clients):
        client_cmd = (
            [sys.executable, "main_client.py", "client"]
            + common_args
            + mode_args
            + client_benchmark_args
            + ["--id_client", str(cid)]
        )
        print(f"Starting client {cid}... (in background)")

        client_log_path = f"{result_dir}/client_{cid}.log"
        client_log_paths.append(client_log_path)
        client_log_file = open(client_log_path, "w")

        # Ensure each client connects to the same server address
        env_client = os.environ.copy()
        env_client["FL_SERVER_ADDRESS"] = server_addr
        client_proc = subprocess.Popen(
            client_cmd,
            stdout=client_log_file,
            stderr=subprocess.STDOUT,
            env=env_client,
        )
        client_procs.append((client_proc, client_log_file, cid))

    print(f"✓ All {num_clients} clients started, waiting for completion...")

    # Wait for all clients to complete
    client_failures = 0
    for client_proc, client_log_file, cid in client_procs:
        try:
            exit_code = client_proc.wait(timeout=600)  # 10 minutes per client
            client_log_file.close()

            if exit_code != 0:
                print(f"⚠️  Client {cid} failed with exit code {exit_code}")
                client_failures += 1
            else:
                print(f"✓ Client {cid} completed")
        except subprocess.TimeoutExpired:
            print(f"⏱️  Client {cid} timed out!")
            client_proc.kill()
            client_log_file.close()
            client_failures += 1
        except Exception as e:
            print(f"❌ Client {cid} error: {e}")
            client_log_file.close()
            client_failures += 1

    # Wait for server to complete (it should exit after all rounds)
    print(f"\nWaiting for server to complete...")
    try:
        server_proc.wait(timeout=600)  # Increased to 10 minutes
        print(f"✓ Server completed")
    except subprocess.TimeoutExpired:
        print(f"⏱️  Server timeout! Terminating...")
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        else:
            server_proc.terminate()
        server_proc.wait(timeout=10)

    # Load benchmark results (in non-simulation, aggregate from separate files)
    benchmark_file = f"{result_dir}/benchmark.json"
    benchmark = None

    # Check if combined benchmark exists (simulation mode) or need to aggregate (non-simulation)
    if os.path.exists(benchmark_file):
        with open(benchmark_file, "r") as f:
            benchmark = json.load(f)
    else:
        # Non-simulation: aggregate client benchmarks
        print("Aggregating client benchmarks...")
        client_benchmarks = []
        for i in range(num_clients):
            client_bench_file = f"{result_dir}/client_{i}_benchmark.json"
            if os.path.exists(client_bench_file):
                with open(client_bench_file, "r") as f:
                    client_benchmarks.append(json.load(f))

        if client_benchmarks:
            # Create aggregated benchmark (simple merge for now)
            benchmark = client_benchmarks[0]  # Use first client as base
            # TODO: Proper aggregation of metrics across clients
            print(f"✓ Aggregated {len(client_benchmarks)} client benchmarks")

    success = (
        server_proc.returncode == 0 and client_failures == 0 and benchmark is not None
    )

    print(f"{display_mode.upper()} {'✓ SUCCESS' if success else '✗ FAILED'}")
    if client_failures > 0:
        print(f"  ({client_failures}/{num_clients} clients failed)")

    return {
        "mode": display_mode,
        "success": success,
        "exit_code": server_proc.returncode,
        "result_dir": result_dir,
        "benchmark": benchmark,
        "client_failures": client_failures,
    }


def run_experiment(mode, base_args, output_dir, he_backend=None, use_simulation=True):
    """Run a single experiment in simulation or non-simulation mode."""
    display_mode = mode if not he_backend else f"{mode}_{he_backend}"
    print(f"\n{'='*60}")
    print(
        f"Running {display_mode.upper()} experiment ({'SIMULATION' if use_simulation else 'NON-SIMULATION'})"
    )
    print(f"{'='*60}\n")

    result_dir = f"{output_dir}/{display_mode}"
    os.makedirs(result_dir, exist_ok=True)

    # Build command - use sys.executable to get current Python
    import sys

    if use_simulation:
        cmd = [sys.executable, "simulation.py", "simulation"]
    else:
        # Non-simulation mode: will start server and clients separately
        return run_experiment_distributed(
            mode, base_args, output_dir, result_dir, he_backend, display_mode
        )

    # Add base arguments
    for key, value in base_args.items():
        if value is not None and value != "":
            cmd.extend([f"--{key}", str(value)])

    # Mode-specific flags
    if mode == "he":
        cmd.append("--he")
        if he_backend:
            cmd.extend(["--he_backend", he_backend])
        cmd.extend(["--path_keys", "secret.pkl"])
        cmd.extend(["--path_public_key", "server_key.pkl"])
    elif mode == "zkp":
        cmd.append("--zkp")
        cmd.extend(["--zkp_params", "zkp_params.pkl"])
    elif mode == "dp":
        cmd.append("--dp")
        cmd.extend(["--dp_params", "dp_params.pkl"])

    # Benchmarking
    cmd.append("--benchmark")
    cmd.extend(["--save_results", f"{result_dir}/"])
    cmd.extend(["--model_save", f"{result_dir}/model.pt"])

    print(f"Command: {' '.join(cmd)}\n")

    # Run
    start_time = datetime.now()
    env = os.environ.copy()
    env["FL_SIMULATION"] = "1"
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
    duration = (datetime.now() - start_time).total_seconds()

    # Save logs
    with open(f"{result_dir}/stdout.log", "w") as f:
        f.write(result.stdout)
    with open(f"{result_dir}/stderr.log", "w") as f:
        f.write(result.stderr)

    # Load benchmark
    benchmark_file = f"{result_dir}/benchmark.json"
    benchmark = None
    if os.path.exists(benchmark_file):
        with open(benchmark_file, "r") as f:
            benchmark = json.load(f)
            # Check if benchmark has actual data
            if benchmark and "timing" in benchmark:
                if benchmark["timing"]["client_fit"]["total"] == 0:
                    print(f"⚠️  Warning: {mode} benchmark shows zero values")
                    benchmark = None

    success = result.returncode == 0 and benchmark is not None
    print(
        f"{display_mode.upper()} {'✓ SUCCESS' if success else '✗ FAILED'} ({duration:.1f}s)"
    )

    return {
        "mode": display_mode,
        "duration": duration,
        "exit_code": result.returncode,
        "result_dir": result_dir,
        "benchmark": benchmark,
        "success": success,
    }


def create_plots(results, output_dir):
    """Create comparison visualizations."""
    valid = [r for r in results if r["success"]]
    if len(valid) < 2:
        print("⚠️  Need at least 2 successful experiments to plot")
        return

    modes = [r["mode"] for r in valid]
    benchmarks = [r["benchmark"] for r in valid]
    colors = {
        "baseline": "#2ecc71",
        "he": "#e74c3c",
        "he_tenseal": "#e74c3c",
        "he_concrete": "#9b59b6",
        "zkp": "#3498db",
        "dp": "#f39c12",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Federated Learning: Baseline vs HE vs ZKP vs DP",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Total Time
    ax = axes[0, 0]
    times = [
        b["timing"]["client_fit"]["total"] + b["timing"]["server_aggregate"]["total"]
        for b in benchmarks
    ]
    bars = ax.bar(modes, times, color=[colors[m] for m in modes], alpha=0.7)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Total Training Time", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}s",
            ha="center",
            va="bottom",
        )

    # 2. Per-Round Time
    ax = axes[0, 1]
    fit_times = [b["timing"]["client_fit"]["mean"] for b in benchmarks]
    bars = ax.bar(modes, fit_times, color=[colors[m] for m in modes], alpha=0.7)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Avg Client Training Time", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, fit_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}s",
            ha="center",
            va="bottom",
        )

    # 3. Communication
    ax = axes[1, 0]
    upload = [
        b["communication_bytes"]["upload"]["total"] / (1024 * 1024) for b in benchmarks
    ]
    download = [
        b["communication_bytes"]["download"]["total"] / (1024 * 1024)
        for b in benchmarks
    ]
    x = np.arange(len(modes))
    width = 0.35
    ax.bar(x - width / 2, upload, width, label="Upload", alpha=0.7)
    ax.bar(x + width / 2, download, width, label="Download", alpha=0.7)
    ax.set_ylabel("Data (MB)", fontsize=12)
    ax.set_title("Communication Overhead", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4. Crypto Overhead
    ax = axes[1, 1]
    crypto_times = []
    labels = []
    for mode, b in zip(modes, benchmarks):
        if "he" in mode and "encryption" in b["timing"]:
            t = b["timing"]["encryption"]["total"] + b["timing"]["decryption"]["total"]
            crypto_times.append(t)
            labels.append(mode.upper().replace("_", "\n"))
        elif mode == "zkp" and "proof_generation" in b["timing"]:
            t = (
                b["timing"]["proof_generation"]["total"]
                + b["timing"]["proof_verification"]["total"]
            )
            crypto_times.append(t)
            labels.append("ZKP")
        elif mode == "dp" and "dp_noise_addition" in b["timing"]:
            t = b["timing"]["dp_noise_addition"]["total"]
            crypto_times.append(t)
            labels.append("DP")
        else:
            crypto_times.append(0)
            labels.append(mode.capitalize())

    bars = ax.bar(labels, crypto_times, color=[colors[m] for m in modes], alpha=0.7)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Cryptographic Overhead", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, crypto_times):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}s",
                ha="center",
                va="bottom",
            )

    # 5. Accuracy
    ax = axes[0, 2]
    accs = []
    for b in benchmarks:
        acc = 0
        if "model_quality" in b:
            mq = b["model_quality"]
            # Try multiple sources (simulation vs non-simulation)
            if mq.get("global_val_accuracy", {}).get("total", 0) > 0:
                acc = mq["global_val_accuracy"]["max"]
            elif mq.get("test_accuracy", {}).get("total", 0) > 0:
                acc = mq["test_accuracy"]["max"]
            elif mq.get("val_accuracy", {}).get("total", 0) > 0:
                acc = mq["val_accuracy"]["max"]
        accs.append(acc)

    bars = ax.bar(modes, accs, color=[colors[m] for m in modes], alpha=0.7)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Best Test/Val Accuracy", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, accs):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}%",
                ha="center",
                va="bottom",
            )

    # 6. Loss
    ax = axes[1, 2]
    # Try to find loss data from any available source
    loss_data = []
    for b in benchmarks:
        if "model_quality" in b:
            mq = b["model_quality"]
            if mq.get("global_val_loss", {}).get("total", 0) > 0:
                loss_data.append(
                    {
                        "initial": mq["global_val_loss"]["max"],
                        "final": mq["global_val_loss"]["min"],
                    }
                )
            elif mq.get("test_loss", {}).get("total", 0) > 0:
                loss_data.append(
                    {"initial": mq["test_loss"]["max"], "final": mq["test_loss"]["min"]}
                )
            elif mq.get("val_loss", {}).get("total", 0) > 0:
                loss_data.append(
                    {"initial": mq["val_loss"]["max"], "final": mq["val_loss"]["min"]}
                )
            else:
                loss_data.append(None)
        else:
            loss_data.append(None)

    has_loss = any(ld is not None for ld in loss_data)
    if has_loss:
        initial = [ld["initial"] if ld else 0 for ld in loss_data]
        final = [ld["final"] if ld else 0 for ld in loss_data]

        x = np.arange(len(modes))
        width = 0.35
        ax.bar(
            x - width / 2, initial, width, label="Initial", alpha=0.7, color="#e74c3c"
        )
        ax.bar(x + width / 2, final, width, label="Final", alpha=0.7, color="#2ecc71")
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Model Convergence", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = f"{output_dir}/comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Plot saved: {plot_path}")
    plt.close()


def print_summary(results):
    """Print results table."""
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")

    print(
        f"{'Mode':<10} {'Status':<10} {'Time(s)':<10} {'Fit(s)':<10} {'Crypto(s)':<12} {'Acc(%)':<10}"
    )
    print("-" * 80)

    for r in results:
        status = "✓" if r["success"] else "✗"
        mode = r["mode"].upper()

        if r["success"]:
            b = r["benchmark"]
            total = (
                b["timing"]["client_fit"]["total"]
                + b["timing"]["server_aggregate"]["total"]
            )
            fit = b["timing"]["client_fit"]["mean"]

            # Crypto
            if r["mode"] == "he" and "encryption" in b["timing"]:
                crypto = (
                    b["timing"]["encryption"]["total"]
                    + b["timing"]["decryption"]["total"]
                )
            elif r["mode"] == "zkp" and "proof_generation" in b["timing"]:
                crypto = (
                    b["timing"]["proof_generation"]["total"]
                    + b["timing"]["proof_verification"]["total"]
                )
            elif r["mode"] == "dp" and "dp_noise_addition" in b["timing"]:
                crypto = b["timing"]["dp_noise_addition"]["total"]
            else:
                crypto = 0

            # Accuracy - try multiple sources
            acc = 0
            if "model_quality" in b:
                mq = b["model_quality"]
                # First try global_val_accuracy (server-side evaluation in simulation)
                if mq.get("global_val_accuracy", {}).get("total", 0) > 0:
                    acc = mq["global_val_accuracy"]["max"]
                # Fallback to test_accuracy (client-side in non-simulation)
                elif mq.get("test_accuracy", {}).get("total", 0) > 0:
                    acc = mq["test_accuracy"]["max"]
                # Fallback to val_accuracy (client validation)
                elif mq.get("val_accuracy", {}).get("total", 0) > 0:
                    acc = mq["val_accuracy"]["max"]

            print(
                f"{mode:<10} {status:<10} {total:<10.1f} {fit:<10.3f} {crypto:<12.1f} {acc:<10.1f}"
            )
        else:
            print(
                f"{mode:<10} {status:<10} {'FAILED':<10} {'-':<10} {'-':<12} {'-':<10}"
            )

    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs HE vs ZKP vs DP")
    parser.add_argument(
        "--modes",
        type=str,
        default="baseline,zkp,dp",
        help="Modes to compare (default: baseline,zkp,dp). Use 'he_tenseal,he_concrete' to compare HE backends.",
    )
    parser.add_argument(
        "--no-simulation",
        action="store_true",
        help="Run in non-simulation mode with real gRPC server/client (default: use simulation mode)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/comparison_all_modes"
    )
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--dataset", type=str, default="cifar")
    parser.add_argument("--number_clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]

    print(f"\n{'='*80}")
    print("FEDERATED LEARNING COMPARISON")
    print(f"{'='*80}")
    print(f"Modes: {', '.join([m.upper() for m in modes])}")
    print(
        f"Mode: {'NON-SIMULATION (Real gRPC)' if args.no_simulation else 'SIMULATION (Ray)'}"
    )
    print(
        f"Config: {args.number_clients} clients, {args.rounds} rounds, {args.max_epochs} epochs/round"
    )
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}")

    # Base arguments
    base_args = {
        "data_path": args.data_path,
        "dataset": args.dataset,
        "number_clients": args.number_clients,
        "rounds": args.rounds,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "min_fit_clients": args.number_clients,
        "min_avail_clients": args.number_clients,
        "min_eval_clients": max(1, args.number_clients // 2),
        "frac_fit": 1.0,
        "frac_eval": 0.5,
    }

    # Check prerequisites
    processed_modes = []
    for mode in modes:
        # Handle HE backend variants
        if mode.startswith("he_"):
            backend = mode.split("_")[1]  # tenseal or concrete
            if not (os.path.exists("secret.pkl") and os.path.exists("server_key.pkl")):
                print(f"\n⚠️  HE mode requires keys. Please run: python create_keys.py")
                print(f"Skipping {mode}...")
                continue
            processed_modes.append(("he", backend))
        elif mode == "he":
            if not (os.path.exists("secret.pkl") and os.path.exists("server_key.pkl")):
                print("\n⚠️  HE mode requires keys. Please run: python create_keys.py")
                print("Skipping HE mode...")
                continue
            processed_modes.append(("he", "tenseal"))  # default to tenseal
        elif mode == "zkp":
            if not os.path.exists("zkp_params.pkl"):
                print(
                    "\n⚠️  ZKP mode requires params. Please run: python create_zkp_params.py"
                )
                print("Skipping ZKP mode...")
                continue
            processed_modes.append((mode, None))
        elif mode == "dp":
            if not os.path.exists("dp_params.pkl"):
                print(
                    "\n⚠️  DP mode requires params. Please run: python create_dp_params.py"
                )
                print("Skipping DP mode...")
                continue
            processed_modes.append((mode, None))
        else:  # baseline
            processed_modes.append((mode, None))

    if len(processed_modes) == 0:
        print("\n❌ No valid modes to run!")
        return

    # Run experiments
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    use_simulation = not args.no_simulation

    for mode, he_backend in processed_modes:
        try:
            result = run_experiment(
                mode, base_args, args.output_dir, he_backend, use_simulation
            )
            results.append(result)
        except subprocess.TimeoutExpired:
            display_name = mode if not he_backend else f"{mode}_{he_backend}"
            print(f"⏱️  {display_name.upper()} timeout!")
            results.append({"mode": display_name, "success": False, "exit_code": -1})
        except Exception as e:
            display_name = mode if not he_backend else f"{mode}_{he_backend}"
            print(f"❌ {display_name.upper()} error: {e}")
            results.append({"mode": display_name, "success": False, "exit_code": -1})

    # Generate outputs
    if any(r["success"] for r in results):
        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": base_args,
            "results": results,
            "summary": {
                "total": len(results),
                "successful": sum(1 for r in results if r["success"]),
                "failed": sum(1 for r in results if not r["success"]),
            },
        }

        with open(f"{args.output_dir}/comparison_report.json", "w") as f:
            json.dump(report, f, indent=2)

        create_plots(results, args.output_dir)
        print_summary(results)

        print(f"{'='*80}")
        print(f"✅ Comparison complete! Results: {args.output_dir}")
        print(f"{'='*80}\n")
    else:
        print("\n❌ All experiments failed!")


if __name__ == "__main__":
    main()
