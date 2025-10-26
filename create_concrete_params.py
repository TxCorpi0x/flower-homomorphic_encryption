#!/usr/bin/env python3
"""
Generate Concrete-ML FHE configuration parameters.

This script creates configuration files for Concrete-ML based FHE inference,
supporting various privacy-accuracy tradeoffs.
"""

import argparse
import sys
from pathlib import Path

try:
    from core.security import (
        ConcreteMLConfig,
        save_concrete_config,
        CONCRETE_ML_AVAILABLE,
    )
except ImportError:
    print("Error: Could not import security module")
    print("Make sure you're running from the project root")
    sys.exit(1)


def create_concrete_params(
    params_path: str = "concrete_params.pkl",
    n_bits: int = 8,
    p_error: float = 0.01,
    rounding_threshold_bits: int = None,
    verbose: bool = False,
):
    """
    Create and save Concrete-ML configuration parameters.

    Args:
        params_path: Output file path
        n_bits: Quantization bit-width (2-8, lower = faster but less accurate)
        p_error: Error probability for FHE operations (lower = more secure but slower)
        rounding_threshold_bits: Rounding optimization parameter
        verbose: Enable verbose output
    """
    if not CONCRETE_ML_AVAILABLE:
        print("ERROR: concrete-ml is not installed!")
        print("Install with: pip install concrete-ml")
        sys.exit(1)

    print("=" * 70)
    print("Concrete-ML FHE Configuration Generator")
    print("=" * 70)

    # Validate parameters
    if not (2 <= n_bits <= 16):
        print(f"Warning: n_bits={n_bits} is unusual. Typical range: 2-8")

    if not (0.0 < p_error <= 0.1):
        print(f"Warning: p_error={p_error} is unusual. Typical range: 0.001-0.1")

    # Create configuration
    config = ConcreteMLConfig(
        n_bits=n_bits,
        rounding_threshold_bits=rounding_threshold_bits,
        p_error=p_error,
        verbose=verbose,
        enable_unsafe_features=False,  # Production: always False
        use_insecure_key_cache=False,  # Production: always False
    )

    # Display configuration
    print("\nConfiguration:")
    print(f"  Quantization bits (n_bits):        {config.n_bits}")
    print(f"  Error probability (p_error):       {config.p_error}")
    print(
        f"  Rounding threshold bits:           {config.rounding_threshold_bits or 'Auto'}"
    )
    print(f"  Verbose:                           {config.verbose}")

    # Privacy-performance tradeoff guidance
    print("\nExpected Performance:")
    if n_bits <= 4:
        print("  ⚡ Very Fast inference (~100ms per sample)")
        print("  📉 Lower accuracy (may lose 5-15% accuracy)")
    elif n_bits <= 6:
        print("  🚀 Fast inference (~500ms per sample)")
        print("  📊 Good accuracy (may lose 2-5% accuracy)")
    else:
        print("  🐢 Slower inference (~1-5s per sample)")
        print("  ✅ Best accuracy (minimal loss <2%)")

    if p_error >= 0.05:
        print("  ⚠️  Higher error tolerance (faster but less reliable)")
    elif p_error >= 0.01:
        print("  ⚖️  Balanced error tolerance")
    else:
        print("  🔒 Low error tolerance (slower but more reliable)")

    # Save configuration
    save_concrete_config(config, params_path)

    print(f"\n✅ Concrete-ML parameters saved to: {params_path}")
    print("\nNext steps:")
    print("  1. Test compilation (optional, requires model):")
    print(
        "     python -c 'from core.security import compile_model_for_fhe, ConcreteMLConfig; from core.model_builder import build_model; import numpy as np; model = build_model(); config = ConcreteMLConfig(n_bits=8); input_sample = np.random.randn(1, 3, 32, 32); compile_model_for_fhe(model, input_sample, config)'"
    )
    print("  2. Use in FL with --concrete flag (when implemented):")
    print(
        "     python simulation.py simulation --concrete --concrete_params concrete_params.pkl"
    )
    print("  3. See core/security.py for FHE compilation and deployment functions")
    print("  4. Refer to Zama's use case examples in tmp/use_case_examples/")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate Concrete-ML FHE configuration parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard configuration (8-bit, balanced)
  python create_concrete_params.py
  
  # Fast inference (6-bit, higher error tolerance)
  python create_concrete_params.py --n_bits 6 --p_error 0.05
  
  # High accuracy (8-bit, low error)
  python create_concrete_params.py --n_bits 8 --p_error 0.001
  
  # Very fast (4-bit, higher error)
  python create_concrete_params.py --n_bits 4 --p_error 0.1
  
  # Custom output path
  python create_concrete_params.py --params_path ./configs/concrete_8bit.pkl

Performance Guide:
  n_bits=2-4:  Very fast (~100ms), lower accuracy
  n_bits=5-6:  Fast (~500ms), good accuracy
  n_bits=7-8:  Slower (~1-5s), best accuracy
  
  p_error=0.001-0.01:  Low error, slower, more secure
  p_error=0.01-0.05:   Balanced
  p_error=0.05-0.1:    High error, faster, less secure
        """,
    )

    parser.add_argument(
        "--params_path",
        type=str,
        default="concrete_params.pkl",
        help="Output file path for parameters (default: concrete_params.pkl)",
    )

    parser.add_argument(
        "--n_bits",
        type=int,
        default=8,
        help="Number of bits for quantization, 2-16 (default: 8, typical: 6-8)",
    )

    parser.add_argument(
        "--p_error",
        type=float,
        default=0.01,
        help="Probability of error for FHE operations (default: 0.01, typical: 0.001-0.1)",
    )

    parser.add_argument(
        "--rounding_threshold_bits",
        type=int,
        default=None,
        help="Rounding threshold for PBS optimization (default: None/auto)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose compilation output"
    )

    args = parser.parse_args()

    try:
        create_concrete_params(
            params_path=args.params_path,
            n_bits=args.n_bits,
            p_error=args.p_error,
            rounding_threshold_bits=args.rounding_threshold_bits,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
