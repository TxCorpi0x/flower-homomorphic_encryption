"""
Script to create Zero-Knowledge Proof parameters for federated learning.
"""
import os
import argparse
from core.zkp import create_zkp_context, write_zkp_params


def generate_zkp_params(params_path="zkp_params.pkl", bit_length=2048):
    """
    Generate and save ZKP parameters.
    
    Args:
        params_path: Path to save the ZKP parameters
        bit_length: Security parameter in bits (default: 2048)
    """
    print(f"Generating ZKP parameters with {bit_length}-bit security...")
    print("This may take a moment...")
    
    # Create ZKP context
    context = create_zkp_context(bit_length=bit_length)
    
    # Save parameters (same params for both client and server in ZKP)
    write_zkp_params(params_path, context)
    
    print(f"✓ ZKP parameters saved to: {params_path}")
    print(f"  - Prime modulus p: {bit_length} bits")
    print(f"  - Group order q: ~{bit_length} bits")
    print("\nYou can now use these parameters with --zkp --zkp_params {params_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ZKP parameters for FL')
    parser.add_argument('--params_path', type=str, default="zkp_params.pkl",
                       help='Path to save ZKP parameters (default: zkp_params.pkl)')
    parser.add_argument('--bit_length', type=int, default=2048,
                       help='Security parameter in bits (default: 2048)')
    
    args = parser.parse_args()
    
    if os.path.exists(args.params_path):
        print(f"Warning: {args.params_path} already exists.")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            exit(0)
    
    generate_zkp_params(args.params_path, args.bit_length)
