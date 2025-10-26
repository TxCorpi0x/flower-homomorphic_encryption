#!/usr/bin/env python3
"""
Aggregate non-simulation results produced by Docker/compose runs.

Scans per-mode result folders (e.g., ./results/baseline, ./results/he_tenseal, ...),
loads per-client benchmark files (client_<N>_benchmark.json), and produces:
  - comparison_report.json
  - comparison.png (summary plot)

Usage:
  python scripts/aggregate_results.py \
    --root ./results \
    --modes baseline,he_tenseal,zkp,dp \
    --output_dir ./results/docker_compare
"""

import argparse
import json
import os
from datetime import datetime

try:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    _PLOTTING_AVAILABLE = True
except Exception:
    plt = None
    np = None
    _PLOTTING_AVAILABLE = False


def _load_mode_benchmark(mode_dir: str):
    # Prefer aggregated benchmark.json if present
    bench_path = os.path.join(mode_dir, "benchmark.json")
    if os.path.exists(bench_path):
        with open(bench_path, "r") as f:
            return json.load(f)

    # Else pick first available client benchmark (simple proxy)
    client_files = sorted(
        [
            os.path.join(mode_dir, f)
            for f in os.listdir(mode_dir)
            if f.startswith("client_") and f.endswith("_benchmark.json")
        ]
    )
    if not client_files:
        return None
    with open(client_files[0], "r") as f:
        return json.load(f)


def _crypto_overhead(mode_name: str, b: dict) -> float:
    t = 0.0
    timing = b.get("timing", {})
    if mode_name.startswith("he"):
        t += timing.get("encryption", {}).get("total", 0) + timing.get(
            "decryption", {}
        ).get("total", 0)
    elif mode_name == "zkp":
        t += timing.get("proof_generation", {}).get("total", 0) + timing.get(
            "proof_verification", {}
        ).get("total", 0)
    elif mode_name == "dp":
        t += timing.get("dp_noise_addition", {}).get("total", 0)
    return float(t)


def _best_accuracy(b: dict) -> float:
    acc = 0.0
    mq = b.get("model_quality", {})
    if mq.get("global_val_accuracy", {}).get("total", 0) > 0:
        acc = mq["global_val_accuracy"].get("max", 0)
    elif mq.get("test_accuracy", {}).get("total", 0) > 0:
        acc = mq["test_accuracy"].get("max", 0)
    elif mq.get("val_accuracy", {}).get("total", 0) > 0:
        acc = mq["val_accuracy"].get("max", 0)
    return float(acc)


def _create_plots(results: list, output_dir: str):
    if not _PLOTTING_AVAILABLE:
        print("Plotting dependencies not available; skipping plot generation.")
        return
    valid = [r for r in results if r["success"]]
    if len(valid) < 1:
        print("No successful modes to plot.")
        return

    modes = [r["mode"] for r in valid]
    benches = [r["benchmark"] for r in valid]

    colors = {
        "baseline": "#2ecc71",
        "he_tenseal": "#e74c3c",
        "he_concrete": "#9b59b6",
        "zkp": "#3498db",
        "dp": "#f39c12",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Federated Learning (Docker): Baseline vs HE vs ZKP vs DP",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Total time (client_fit + server_aggregate if present)
    ax = axes[0, 0]
    times = [
        b.get("timing", {}).get("client_fit", {}).get("total", 0)
        + b.get("timing", {}).get("server_aggregate", {}).get("total", 0)
        for b in benches
    ]
    bars = ax.bar(modes, times, color=[colors.get(m, "gray") for m in modes], alpha=0.7)
    ax.set_ylabel("Time (s)")
    ax.set_title("Total Training Time")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}s",
            ha="center",
            va="bottom",
        )

    # 2. Avg client fit time
    ax = axes[0, 1]
    fit_mean = [
        b.get("timing", {}).get("client_fit", {}).get("mean", 0) for b in benches
    ]
    bars = ax.bar(
        modes, fit_mean, color=[colors.get(m, "gray") for m in modes], alpha=0.7
    )
    ax.set_ylabel("Time (s)")
    ax.set_title("Avg Client Training Time")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, fit_mean):
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
        b.get("communication_bytes", {}).get("upload", {}).get("total", 0)
        / (1024 * 1024)
        for b in benches
    ]
    download = [
        b.get("communication_bytes", {}).get("download", {}).get("total", 0)
        / (1024 * 1024)
        for b in benches
    ]
    x = np.arange(len(modes))
    width = 0.35
    ax.bar(x - width / 2, upload, width, label="Upload", alpha=0.7)
    ax.bar(x + width / 2, download, width, label="Download", alpha=0.7)
    ax.set_ylabel("Data (MB)")
    ax.set_title("Communication Overhead")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4. Crypto overhead
    ax = axes[1, 1]
    crypto = []
    labels = []
    for m, b in zip(modes, benches):
        crypto.append(_crypto_overhead(m, b))
        labels.append(m.upper().replace("_", "\n"))
    bars = ax.bar(
        labels, crypto, color=[colors.get(m, "gray") for m in modes], alpha=0.7
    )
    ax.set_ylabel("Time (s)")
    ax.set_title("Cryptographic Overhead")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, crypto):
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
    accs = [_best_accuracy(b) for b in benches]
    bars = ax.bar(modes, accs, color=[colors.get(m, "gray") for m in modes], alpha=0.7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Best Test/Val Accuracy")
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

    # 6. Loss (initial vs final if available)
    ax = axes[1, 2]
    loss_data = []
    for b in benches:
        mq = b.get("model_quality", {})
        if mq.get("global_val_loss", {}).get("total", 0) > 0:
            loss_data.append(
                {
                    "initial": mq["global_val_loss"].get("max", 0),
                    "final": mq["global_val_loss"].get("min", 0),
                }
            )
        elif mq.get("test_loss", {}).get("total", 0) > 0:
            loss_data.append(
                {
                    "initial": mq["test_loss"].get("max", 0),
                    "final": mq["test_loss"].get("min", 0),
                }
            )
        elif mq.get("val_loss", {}).get("total", 0) > 0:
            loss_data.append(
                {
                    "initial": mq["val_loss"].get("max", 0),
                    "final": mq["val_loss"].get("min", 0),
                }
            )
        else:
            loss_data.append(None)

    if any(ld is not None for ld in loss_data):
        initial = [ld["initial"] if ld else 0 for ld in loss_data]
        final = [ld["final"] if ld else 0 for ld in loss_data]

        x = np.arange(len(modes))
        width = 0.35
        ax.bar(
            x - width / 2, initial, width, label="Initial", alpha=0.7, color="#e74c3c"
        )
        ax.bar(x + width / 2, final, width, label="Final", alpha=0.7, color="#2ecc71")
        ax.set_ylabel("Loss")
        ax.set_title("Model Convergence")
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Plot saved: {plot_path}")
    plt.close()


def _print_summary(results: list):
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (Docker Aggregation)")
    print("=" * 80 + "\n")

    print(
        f"{'Mode':<12} {'Status':<10} {'Time(s)':<10} {'Fit(s)':<10} {'Crypto(s)':<12} {'Acc(%)':<10}"
    )
    print("-" * 80)

    for r in results:
        status = "✓" if r["success"] else "✗"
        mode = r["mode"].upper()
        if r["success"]:
            b = r["benchmark"]
            timing = b.get("timing", {})
            total = timing.get("client_fit", {}).get("total", 0) + timing.get(
                "server_aggregate", {}
            ).get("total", 0)
            fit = timing.get("client_fit", {}).get("mean", 0)
            crypto = _crypto_overhead(r["mode"], b)
            acc = _best_accuracy(b)
            print(
                f"{mode:<12} {status:<10} {total:<10.1f} {fit:<10.3f} {crypto:<12.1f} {acc:<10.1f}"
            )
        else:
            print(f"{mode:<12} {status:<10} {'-':<10} {'-':<10} {'-':<12} {'-':<10}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Aggregate Docker non-simulation results")
    ap.add_argument(
        "--root",
        type=str,
        default="./results",
        help="Results root containing per-mode subfolders",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="",
        help="Comma-separated modes to include (auto-detect by default)",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="./results/docker_compare",
        help="Output folder for report and plot",
    )
    args = ap.parse_args()

    if args.modes:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    else:
        # Auto-detect subfolders containing client_*_benchmark.json
        modes = []
        if not os.path.isdir(args.root):
            print(f"No results root found: {args.root}")
            return
        for name in os.listdir(args.root):
            mode_dir = os.path.join(args.root, name)
            if not os.path.isdir(mode_dir):
                continue
            if any(
                fn.startswith("client_") and fn.endswith("_benchmark.json")
                for fn in os.listdir(mode_dir)
            ):
                modes.append(name)

    if not modes:
        print("No modes found to aggregate.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for mode in modes:
        mode_dir = os.path.join(args.root, mode)
        if not os.path.isdir(mode_dir):
            print(f"Skipping '{mode}' - directory not found: {mode_dir}")
            continue
        bench = _load_mode_benchmark(mode_dir)
        success = bench is not None and "timing" in bench
        results.append(
            {
                "mode": mode,
                "success": success,
                "benchmark": bench if bench else {},
                "result_dir": mode_dir,
            }
        )

    if not results:
        print("No modes found to aggregate.")
        return

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {"root": args.root, "modes": modes},
        "results": results,
        "summary": {
            "total": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
        },
    }
    with open(os.path.join(args.output_dir, "comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report: {os.path.join(args.output_dir, 'comparison_report.json')}")

    _create_plots(results, args.output_dir)
    _print_summary(results)


if __name__ == "__main__":
    main()
