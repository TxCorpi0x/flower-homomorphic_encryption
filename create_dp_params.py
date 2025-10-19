"""
Create and save Differential Privacy parameters for federated learning.

This script generates DP configuration with reasonable defaults for privacy-utility tradeoff.
"""

import argparse
from core.security import DifferentialPrivacyParams, save_dp_params


def main():
    parser = argparse.ArgumentParser(
        description="Create Differential Privacy parameters for FL"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Privacy budget (smaller = more private, typical: 0.1-10). Default: 1.0",
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Probability of privacy breach (typical: 1e-5). Default: 1e-5",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (typical: 0.1-2.0). Default: 1.0",
    )

    parser.add_argument(
        "--mechanism",
        type=str,
        default="gaussian",
        choices=["gaussian", "laplace"],
        help="Noise mechanism. Default: gaussian",
    )

    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=None,
        help="Custom noise multiplier (auto-computed if not provided)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dp_params.pkl",
        help="Output file path. Default: dp_params.pkl",
    )

    args = parser.parse_args()

    # Create DP parameters
    dp_params = DifferentialPrivacyParams(
        epsilon=args.epsilon,
        delta=args.delta,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        mechanism=args.mechanism,
    )

    # Save to file
    save_dp_params(dp_params, args.output)

    print(f"\n{'='*60}")
    print("Differential Privacy Parameters Created")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  Epsilon (ε):        {dp_params.epsilon}")
    print(f"  Delta (δ):          {dp_params.delta}")
    print(f"  Max Grad Norm (C):  {dp_params.max_grad_norm}")
    print(f"  Noise Multiplier:   {dp_params.noise_multiplier:.4f}")
    print(f"  Mechanism:          {dp_params.mechanism}")

    print(f"\n{'='*60}")
    print(f"Privacy Interpretation:")
    print(f"{'='*60}")

    if dp_params.epsilon < 0.5:
        privacy_level = "Very Strong"
        utility_impact = "Significant accuracy loss expected"
    elif dp_params.epsilon < 1.0:
        privacy_level = "Strong"
        utility_impact = "Moderate accuracy loss expected"
    elif dp_params.epsilon < 3.0:
        privacy_level = "Moderate"
        utility_impact = "Minimal accuracy loss expected"
    else:
        privacy_level = "Weak"
        utility_impact = "Negligible accuracy loss expected"

    print(f"  Privacy Level:      {privacy_level}")
    print(f"  Utility Impact:     {utility_impact}")
    print(f"  Recommended for:    {get_recommendation(dp_params.epsilon)}")

    print(f"\n{'='*60}")
    print(f"✅ DP parameters saved to: {args.output}")
    print(f"{'='*60}\n")

    print("Usage in simulation:")
    print(f"  python simulation.py simulation --dp --dp_params {args.output}")
    print("\nUsage in comparison:")
    print(f"  python compare_methods_simple.py --modes baseline,he,zkp,dp")


def get_recommendation(epsilon):
    """Get recommendation based on epsilon value."""
    if epsilon < 0.5:
        return "High-security medical/financial data"
    elif epsilon < 1.0:
        return "Sensitive personal data"
    elif epsilon < 3.0:
        return "General purpose privacy protection"
    else:
        return "Compliance requirements only"


if __name__ == "__main__":
    main()
