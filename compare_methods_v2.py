"""
Enhanced comparison script that runs each mode in its optimal way:
- Baseline & ZKP: Use simulation mode (fast, works great)
- HE: Use traditional federated mode (avoids serialization issues)
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


class ExperimentRunner:
    """Handles running experiments in different modes."""

    def __init__(self, base_args, output_dir):
        self.base_args = base_args
        self.output_dir = output_dir
        self.processes = []

    def cleanup(self):
        """Cleanup any running processes."""
        for proc in self.processes:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except:
                    proc.kill()

    def run_simulation_mode(self, mode, result_dir):
        """
        Run baseline or ZKP in simulation mode (works great).

        Args:
            mode: "baseline" or "zkp"
            result_dir: Directory to save results

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"Running {mode.upper()} in SIMULATION mode")
        print(f"{'='*60}\n")
        cmd = [sys.executable, "simulation.py", "simulation"]

        # Add base arguments
        for key, value in self.base_args.items():
            if value is not None and value != "":
                cmd.extend([f"--{key}", str(value)])

        # Mode-specific flags
        if mode == "zkp":
            cmd.append("--zkp")
            cmd.extend(["--zkp_params", "zkp_params.pkl"])

        # Enable benchmarking
        cmd.append("--benchmark")

        # Output paths
        os.makedirs(result_dir, exist_ok=True)
        cmd.extend(["--save_results", f"{result_dir}/"])
        cmd.extend(["--model_save", f"{result_dir}/model.pt"])

        print(f"Command: {' '.join(cmd)}\n")

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time

        # Save logs
        with open(f"{result_dir}/stdout.log", "w") as f:
            f.write(result.stdout)
        with open(f"{result_dir}/stderr.log", "w") as f:
            f.write(result.stderr)

        # Load benchmark
        benchmark_file = f"{result_dir}/benchmark.json"
        benchmark_data = None
        if os.path.exists(benchmark_file):
            with open(benchmark_file, "r") as f:
                benchmark_data = json.load(f)

        print(
            f"{mode.upper()} completed in {duration:.2f}s (exit code: {result.returncode})"
        )

        return {
            "mode": mode,
            "duration": duration,
            "exit_code": result.returncode,
            "result_dir": result_dir,
            "benchmark": benchmark_data,
            "run_mode": "simulation",
        }

    def run_he_federated_mode(self, result_dir):
        """
        Run HE in traditional federated mode (server + clients).
        This avoids the Flower simulation serialization issues.

        Args:
            result_dir: Directory to save results

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"Running HE in FEDERATED mode (server + clients)")
        print(f"{'='*60}\n")

        os.makedirs(result_dir, exist_ok=True)

        # Use federated.py for HE mode with benchmarking
        server_cmd = [
            sys.executable,
            "main_server.py",
            "server",
            "--data_path",
            str(self.base_args.get("data_path", "./data/")),
            "--dataset",
            str(self.base_args.get("dataset", "cifar")),
            "--device",
            str(self.base_args.get("device", "cpu")),
            "--rounds",
            str(self.base_args.get("rounds", 2)),
            "--min_fit_clients",
            str(self.base_args.get("number_clients", 4)),
            "--min_avail_clients",
            str(self.base_args.get("number_clients", 4)),
            "--he",
            "--benchmark",
            "--model_save",
            f"{result_dir}/model.pt",
        ]

        client_cmd_template = [
            sys.executable,
            "main_client.py",
            "client",
            "--data_path",
            str(self.base_args.get("data_path", "./data/")),
            "--dataset",
            str(self.base_args.get("dataset", "cifar")),
            "--device",
            str(self.base_args.get("device", "cpu")),
            "--max_epochs",
            str(self.base_args.get("max_epochs", 1)),
            "--batch_size",
            str(self.base_args.get("batch_size", 32)),
            "--he",
        ]

        print(f"Server command: {' '.join(server_cmd)}")
        print(f"Client template: {' '.join(client_cmd_template)}")
        print("\nStarting server...")

        # Start server
        server_log = open(f"{result_dir}/server.log", "w")
        server_proc = subprocess.Popen(
            server_cmd, stdout=server_log, stderr=subprocess.STDOUT, text=True
        )
        self.processes.append(server_proc)

        # Wait for server to start
        time.sleep(5)

        # Start clients
        num_clients = self.base_args.get("number_clients", 4)
        client_procs = []

        print(f"\nStarting {num_clients} clients...")
        for client_id in range(num_clients):
            client_log = open(f"{result_dir}/client_{client_id}.log", "w")
            client_proc = subprocess.Popen(
                client_cmd_template
                + ["--id_client", str(client_id), "--save_results", f"{result_dir}/"],
                stdout=client_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            client_procs.append(client_proc)
            self.processes.append(client_proc)
            time.sleep(1)  # Stagger client starts

        print("\nWaiting for training to complete...")
        start_time = time.time()

        # Wait for server to complete
        try:
            server_proc.wait(timeout=600)  # 10 minute timeout
        except subprocess.TimeoutExpired:
            print("⚠️  Server timeout - terminating...")
            server_proc.terminate()

        # Terminate clients
        for proc in client_procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except:
                    proc.kill()

        duration = time.time() - start_time
        server_log.close()

        # Load benchmark if available
        benchmark_file = f"{result_dir}/benchmark.json"
        benchmark_data = None
        if os.path.exists(benchmark_file):
            with open(benchmark_file, "r") as f:
                benchmark_data = json.load(f)

        print(f"HE completed in {duration:.2f}s")

        return {
            "mode": "he",
            "duration": duration,
            "exit_code": server_proc.returncode,
            "result_dir": result_dir,
            "benchmark": benchmark_data,
            "run_mode": "federated",
        }


def plot_comparison(results, output_dir):
    """Create comparison plots."""
    modes = [r["mode"] for r in results]
    colors = {"baseline": "#2ecc71", "he": "#e74c3c", "zkp": "#3498db"}

    benchmarks = [r.get("benchmark") for r in results]

    # Filter out failed experiments
    valid_results = [(m, b) for m, b in zip(modes, benchmarks) if b is not None]
    if not valid_results:
        print("⚠️  No valid benchmark data to plot")
        return

    modes, benchmarks = zip(*valid_results)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Federated Learning Performance Comparison", fontsize=16, fontweight="bold"
    )

    # 1. Total Training Time
    ax = axes[0, 0]
    if all("timing" in b for b in benchmarks):
        timing_data = []
        for b in benchmarks:
            total = (
                b["timing"]["client_fit"]["total"]
                + b["timing"]["server_aggregate"]["total"]
            )
            timing_data.append(total)

        bars = ax.bar(modes, timing_data, color=[colors[m] for m in modes], alpha=0.7)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title("Total Training Time", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, timing_data):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}s",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # 2. Client Fit Time
    ax = axes[0, 1]
    if all("timing" in b for b in benchmarks):
        fit_times = [b["timing"]["client_fit"]["mean"] for b in benchmarks]
        bars = ax.bar(modes, fit_times, color=[colors[m] for m in modes], alpha=0.7)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title("Avg Client Training Time", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, fit_times):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}s",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # 3. Communication Overhead
    ax = axes[1, 0]
    if all("communication_bytes" in b for b in benchmarks):
        upload_mb = [
            b["communication_bytes"]["upload"]["total"] / (1024 * 1024)
            for b in benchmarks
        ]
        download_mb = [
            b["communication_bytes"]["download"]["total"] / (1024 * 1024)
            for b in benchmarks
        ]

        x = np.arange(len(modes))
        width = 0.35
        ax.bar(x - width / 2, upload_mb, width, label="Upload", alpha=0.7)
        ax.bar(x + width / 2, download_mb, width, label="Download", alpha=0.7)

        ax.set_ylabel("Data (MB)", fontsize=12)
        ax.set_title("Communication Overhead", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    # 4. Cryptographic Overhead
    ax = axes[1, 1]
    if all("timing" in b for b in benchmarks):
        crypto_times = []
        labels = []

        for mode, b in zip(modes, benchmarks):
            if mode == "he" and "encryption" in b["timing"]:
                crypto_time = (
                    b["timing"]["encryption"]["total"]
                    + b["timing"]["decryption"]["total"]
                )
                crypto_times.append(crypto_time)
                labels.append("HE")
            elif mode == "zkp" and "proof_generation" in b["timing"]:
                crypto_time = (
                    b["timing"]["proof_generation"]["total"]
                    + b["timing"]["proof_verification"]["total"]
                )
                crypto_times.append(crypto_time)
                labels.append("ZKP")
            else:
                crypto_times.append(0)
                labels.append("Baseline")

        bars = ax.bar(labels, crypto_times, color=[colors[m] for m in modes], alpha=0.7)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title("Cryptographic Overhead", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, crypto_times):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    # 5. Model Accuracy
    ax = axes[0, 2]
    if all(
        "model_quality" in b and b["model_quality"]["global_val_accuracy"]["total"] > 0
        for b in benchmarks
    ):
        accuracies = [
            b["model_quality"]["global_val_accuracy"]["max"] for b in benchmarks
        ]
        bars = ax.bar(modes, accuracies, color=[colors[m] for m in modes], alpha=0.7)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Best Global Accuracy", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # 6. Loss Convergence
    ax = axes[1, 2]
    if all(
        "model_quality" in b and b["model_quality"]["global_val_loss"]["total"] > 0
        for b in benchmarks
    ):
        initial_loss = [
            b["model_quality"]["global_val_loss"]["max"] for b in benchmarks
        ]
        final_loss = [b["model_quality"]["global_val_loss"]["min"] for b in benchmarks]

        x = np.arange(len(modes))
        width = 0.35
        ax.bar(
            x - width / 2,
            initial_loss,
            width,
            label="Initial",
            alpha=0.7,
            color="#e74c3c",
        )
        ax.bar(
            x + width / 2, final_loss, width, label="Final", alpha=0.7, color="#2ecc71"
        )

        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Model Convergence", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = f"{output_dir}/comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Comparison plot saved: {plot_path}")
    plt.close()


def print_summary(results):
    """Print summary table."""
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}\n")

    print(
        f"{'Mode':<12} {'Run Mode':<15} {'Time(s)':<12} {'Fit(s)':<12} {'Crypto(s)':<12} {'Acc(%)':<12}"
    )
    print("-" * 80)

    for r in results:
        b = r.get("benchmark")
        if b and "timing" in b:
            total_time = (
                b["timing"]["client_fit"]["total"]
                + b["timing"]["server_aggregate"]["total"]
            )
            fit_time = b["timing"]["client_fit"]["mean"]

            # Crypto time
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
            else:
                crypto = 0

            # Accuracy
            acc = 0
            if (
                "model_quality" in b
                and b["model_quality"]["global_val_accuracy"]["total"] > 0
            ):
                acc = b["model_quality"]["global_val_accuracy"]["max"]

            print(
                f"{r['mode'].upper():<12} {r['run_mode']:<15} {total_time:<12.2f} {fit_time:<12.3f} {crypto:<12.2f} {acc:<12.2f}"
            )
        else:
            print(
                f"{r['mode'].upper():<12} {r.get('run_mode', 'N/A'):<15} {'FAILED':<12} {'-':<12} {'-':<12} {'-':<12}"
            )

    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare FL modes with optimal execution strategies"
    )

    parser.add_argument(
        "--modes",
        type=str,
        default="baseline,he,zkp",
        help="Comma-separated modes to compare",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/comparison_all_modes",
        help="Output directory",
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
    print("FEDERATED LEARNING COMPARISON - OPTIMIZED EXECUTION")
    print(f"{'='*80}")
    print(f"Modes: {', '.join([m.upper() for m in modes])}")
    print(f"Strategy: Baseline/ZKP use simulation, HE uses federated mode")
    print(f"Clients: {args.number_clients}, Rounds: {args.rounds}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Prepare base arguments
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

    # Create experiment runner
    runner = ExperimentRunner(base_args, args.output_dir)

    # Setup signal handler for cleanup
    def signal_handler(sig, frame):
        print("\n\n⚠️  Interrupted! Cleaning up...")
        runner.cleanup()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    # Check prerequisites
    if "he" in modes and not (
        os.path.exists("secret.pkl") and os.path.exists("server_key.pkl")
    ):
        print("⚠️  HE mode requires keys. Run: python create_keys.py")
        modes.remove("he")

    if "zkp" in modes and not os.path.exists("zkp_params.pkl"):
        print("⚠️  ZKP mode requires params. Run: python create_zkp_params.py")
        modes.remove("zkp")

    # Run experiments
    results = []
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        for mode in modes:
            result_dir = f"{args.output_dir}/{mode}"

            if mode in ["baseline", "zkp"]:
                # Use simulation mode (fast and reliable)
                result = runner.run_simulation_mode(mode, result_dir)
            elif mode == "he":
                # Use federated mode (avoids serialization issues)
                result = runner.run_he_federated_mode(result_dir)
            else:
                print(f"❌ Unknown mode: {mode}")
                continue

            results.append(result)

    finally:
        runner.cleanup()

    # Generate comparison
    if len(results) > 0:
        # Save detailed report
        report_path = f"{args.output_dir}/comparison_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "experiments": results,
                    "summary": {
                        "modes_compared": [r["mode"] for r in results],
                        "successful": sum(
                            1 for r in results if r.get("benchmark") is not None
                        ),
                    },
                },
                f,
                indent=2,
            )

        plot_comparison(results, args.output_dir)
        print_summary(results)

        print(f"{'='*80}")
        print(f"✅ Comparison complete! Results in: {args.output_dir}")
        print(f"{'='*80}\n")
    else:
        print("\n❌ No experiments completed")


if __name__ == "__main__":
    main()
