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


def run_experiment(mode, base_args, output_dir):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Running {mode.upper()} experiment")
    print(f"{'='*60}\n")

    result_dir = f"{output_dir}/{mode}"
    os.makedirs(result_dir, exist_ok=True)

    # Build command - use sys.executable to get current Python
    import sys

    cmd = [sys.executable, "simulation.py", "simulation"]

    # Add base arguments
    for key, value in base_args.items():
        if value is not None and value != "":
            cmd.extend([f"--{key}", str(value)])

    # Mode-specific flags
    if mode == "he":
        cmd.append("--he")
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
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
    print(f"{mode.upper()} {'✓ SUCCESS' if success else '✗ FAILED'} ({duration:.1f}s)")

    return {
        "mode": mode,
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
    colors = {"baseline": "#2ecc71", "he": "#e74c3c", "zkp": "#3498db", "dp": "#f39c12"}

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
        if mode == "he" and "encryption" in b["timing"]:
            t = b["timing"]["encryption"]["total"] + b["timing"]["decryption"]["total"]
            crypto_times.append(t)
            labels.append("HE")
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
        if (
            "model_quality" in b
            and b["model_quality"]["global_val_accuracy"]["total"] > 0
        ):
            accs.append(b["model_quality"]["global_val_accuracy"]["max"])
        else:
            accs.append(0)

    bars = ax.bar(modes, accs, color=[colors[m] for m in modes], alpha=0.7)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Best Global Accuracy", fontsize=13, fontweight="bold")
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
    has_loss = all(
        "model_quality" in b and b["model_quality"]["global_val_loss"]["total"] > 0
        for b in benchmarks
    )
    if has_loss:
        initial = [b["model_quality"]["global_val_loss"]["max"] for b in benchmarks]
        final = [b["model_quality"]["global_val_loss"]["min"] for b in benchmarks]

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

            # Accuracy
            acc = 0
            if (
                "model_quality" in b
                and b["model_quality"]["global_val_accuracy"]["total"] > 0
            ):
                acc = b["model_quality"]["global_val_accuracy"]["max"]

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
        help="Modes to compare (default: baseline,zkp,dp)",
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
    if "he" in modes:
        if not (os.path.exists("secret.pkl") and os.path.exists("server_key.pkl")):
            print("\n⚠️  HE mode requires keys. Please run: python create_keys.py")
            print("Skipping HE mode...")
            modes.remove("he")
        else:
            print("\n⚠️  NOTE: HE mode has known issues in simulation.")
            print(
                "    Results may not be reliable. Consider running baseline and ZKP only."
            )
            response = input("Continue with HE? (y/n): ")
            if response.lower() != "y":
                modes.remove("he")

    if "zkp" in modes and not os.path.exists("zkp_params.pkl"):
        print("\n⚠️  ZKP mode requires params. Please run: python create_zkp_params.py")
        print("Skipping ZKP mode...")
        modes.remove("zkp")

    if "dp" in modes and not os.path.exists("dp_params.pkl"):
        print("\n⚠️  DP mode requires params. Please run: python create_dp_params.py")
        print("Skipping DP mode...")
        modes.remove("dp")

    if len(modes) == 0:
        print("\n❌ No valid modes to run!")
        return

    # Run experiments
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for mode in modes:
        try:
            result = run_experiment(mode, base_args, args.output_dir)
            results.append(result)
        except subprocess.TimeoutExpired:
            print(f"⏱️  {mode.upper()} timeout!")
            results.append({"mode": mode, "success": False, "exit_code": -1})
        except Exception as e:
            print(f"❌ {mode.upper()} error: {e}")
            results.append({"mode": mode, "success": False, "exit_code": -1})

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
