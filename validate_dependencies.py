#!/usr/bin/env python3
"""
Dependency validation script for ZK + Blockchain Federated Learning

This script validates that all required dependencies are properly installed
and the core functionality works as expected.
"""

import sys
import importlib
from typing import List, Tuple


def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Check if a module can be imported"""
    try:
        if package_name:
            module = importlib.import_module(module_name, package_name)
        else:
            module = importlib.import_module(module_name)
        return True, f"✓ {module_name}"
    except ImportError as e:
        return False, f"✗ {module_name}: {str(e)}"


def validate_dependencies():
    """Validate all required dependencies"""

    print("🔍 Validating ZK + Blockchain FL Dependencies")
    print("=" * 60)

    # Core ML and FL dependencies
    core_deps = [
        ("torch", None),
        ("torchvision", None),
        ("numpy", None),
        ("flwr", None),
        ("ray", None),
        ("sklearn", None),
    ]

    # Data processing dependencies
    data_deps = [
        ("matplotlib", None),
        ("matplotlib.pyplot", None),
        ("seaborn", None),
        ("pandas", None),
        ("PIL", None),  # Pillow
        ("tqdm", None),
        ("yaml", None),
    ]

    # Privacy and security dependencies
    security_deps = [
        ("opacus", None),
        ("cryptography", None),
        ("Crypto", None),  # pycryptodome
        ("hashlib", None),
        ("hmac", None),
        ("secrets", None),
    ]

    # Blockchain dependencies
    blockchain_deps = [
        ("web3", None),
        ("eth_account", None),
        ("eth_utils", None),
    ]

    # Development dependencies
    dev_deps = [
        ("pytest", None),
    ]

    all_deps = [
        ("Core ML & FL", core_deps),
        ("Data Processing", data_deps),
        ("Security & Privacy", security_deps),
        ("Blockchain", blockchain_deps),
        ("Development", dev_deps),
    ]

    total_tested = 0
    total_passed = 0

    for category, deps in all_deps:
        print(f"\n📦 {category} Dependencies:")
        category_passed = 0

        for module_name, package_name in deps:
            success, message = check_import(module_name, package_name)
            print(f"   {message}")

            if success:
                category_passed += 1
                total_passed += 1
            total_tested += 1

        print(f"   └─ {category}: {category_passed}/{len(deps)} passed")

    print(f"\n" + "=" * 60)
    print(f"📊 Overall Results: {total_passed}/{total_tested} dependencies available")

    # Test core functionality
    print(f"\n🧪 Testing Core Functionality:")

    try:
        # Test PyTorch
        import torch

        x = torch.randn(2, 3)
        print(f"   ✓ PyTorch tensor operations: {x.shape}")

        # Test our modules
        from going_modular.zk_security import (
            ZKModelProof,
            DifferentialPrivacy,
            BlockchainStorage,
        )

        zk = ZKModelProof()
        print(f"   ✓ ZK Proof system initialized")

        dp = DifferentialPrivacy(noise_scale=0.1)
        print(f"   ✓ Differential Privacy system initialized")

        blockchain = BlockchainStorage()
        print(f"   ✓ Blockchain storage initialized")

        # Test model creation
        from going_modular.model_builder import Net

        model = Net(num_classes=10)
        print(f"   ✓ Neural network model created")

        print(f"\n🎉 All core functionality tests passed!")

    except Exception as e:
        print(f"   ✗ Core functionality test failed: {e}")
        return False

    return total_passed == total_tested


def main():
    """Main validation function"""

    print("Python version:", sys.version)
    print("Python executable:", sys.executable)
    print()

    success = validate_dependencies()

    if success:
        print("\n🚀 Ready to run ZK + Blockchain Federated Learning!")
        print("\nNext steps:")
        print("1. Run demo: python demo_zk_blockchain.py")
        print("2. Run tests: python test_zk_blockchain.py")
        print(
            "3. Run simulation: python simulation.py simulation --enable_zk --differential_privacy --enable_blockchain"
        )
        return 0
    else:
        print("\n⚠️  Some dependencies are missing. Please install them:")
        print("pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
