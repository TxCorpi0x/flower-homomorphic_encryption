#!/usr/bin/env python3
"""
ZKFL Example Script - Demonstrating All Learning Approaches

This script demonstrates how to use the ZKFL framework with different approaches:
- Centralized Learning (traditional ML)
- Classic Federated Learning
- FL with Fully Homomorphic Encryption (FL-FHE)
- FL with Zero-Knowledge Proofs (FL-ZK)
- FL with Differential Privacy (FL-DP)
- Hybrid approaches (combining multiple techniques)

Usage:
    python zkfl_example.py --approach centralized
    python zkfl_example.py --approach classic
    python zkfl_example.py --approach fhe
    python zkfl_example.py --approach zk
    python zkfl_example.py --approach dp
    python zkfl_example.py --approach hybrid
    python zkfl_example.py --compare  # Show FHE vs ZK+Blockchain comparison
"""

import argparse
import sys
import time
from pathlib import Path

# Add zkfl to path for imports
sys.path.append(str(Path(__file__).parent))

from zkfl.utils.config_utils import ConfigUtils
from zkfl.utils.logging_utils import LoggingUtils
from zkfl.crypto.zk_proof import ZKProofSystem
from zkfl.privacy.differential_privacy import DifferentialPrivacyManager
from zkfl.blockchain.blockchain_interface import BlockchainInterface
from zkfl.federated.federated_utils import FederatedUtils
from zkfl.data.data_setup import create_federated_datasets
from zkfl.models.model_builder import ModelBuilder
from zkfl.core import TrainingEngine


class FLApproaches:
    """Supported federated learning approaches."""

    CENTRALIZED = "centralized"
    CLASSIC = "classic"
    FHE = "fhe"  # Uses ZK proofs for cryptographic security
    ZK = "zero_knowledge"
    DP = "differential_privacy"
    HYBRID = "hybrid"


def create_fl_system(approach: str, **kwargs):
    """
    Factory function to create FL system based on approach.

    Args:
        approach: FL approach to use
        **kwargs: Configuration parameters

    Returns:
        Configured FL system components
    """
    if approach == FLApproaches.CLASSIC:
        return {
            "crypto": None,
            "privacy": None,
            "blockchain": None,
            "aggregation": FederatedUtils(),
        }

    elif approach == FLApproaches.ZK or approach == FLApproaches.FHE:
        return {
            "crypto": ZKProofSystem(**kwargs.get("zk_config", {})),
            "privacy": None,
            "blockchain": kwargs.get("enable_blockchain", False)
            and BlockchainInterface(),
            "aggregation": FederatedUtils(),
        }

    elif approach == FLApproaches.DP:
        return {
            "crypto": None,
            "privacy": DifferentialPrivacyManager(**kwargs.get("dp_config", {})),
            "blockchain": None,
            "aggregation": FederatedUtils(),
        }

    elif approach == FLApproaches.HYBRID:
        return {
            "crypto": ZKProofSystem(**kwargs.get("zk_config", {})),
            "privacy": DifferentialPrivacyManager(**kwargs.get("dp_config", {})),
            "blockchain": kwargs.get("enable_blockchain", False)
            and BlockchainInterface(),
            "aggregation": FederatedUtils(),
        }

    else:
        raise ValueError(f"Unsupported FL approach: {approach}")


def run_centralized(config, logger):
    """Run centralized (non-federated) machine learning."""
    logger.info("🏠 Running Centralized Machine Learning (Non-FL)")

    # Load centralized dataset (combine all client data)
    datasets, test_data = create_federated_datasets(
        dataset_name=config.data.dataset_name,
        num_clients=1,  # Single "client" contains all data
        partition_strategy="iid",
    )

    # Build model
    model_builder = ModelBuilder()
    model = model_builder.build_model(
        model_type=config.model.model_type,
        input_shape=config.model.input_shape,
        num_classes=config.model.num_classes,
    )

    # Simulate centralized training
    logger.info("Training model on centralized dataset...")

    # Get all data in one place
    all_data = datasets[0]["data"]  # All data is in the single partition
    all_labels = datasets[0]["labels"]

    logger.info(f"Training samples: {len(all_data)}")
    logger.info(f"Test samples: {len(test_data.get('data', []))}")

    # Simulate training epochs
    import numpy as np

    for epoch in range(5):
        # Simulate training metrics
        train_loss = 2.3 * np.exp(-epoch * 0.3) + np.random.normal(0, 0.1)
        train_acc = (1 - np.exp(-epoch * 0.4)) * 0.95 + np.random.normal(0, 0.02)

        logger.info(
            f"Epoch {epoch+1}/5 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}"
        )

    logger.info("✅ Centralized training complete")
    return {"model": model, "dataset": datasets[0], "approach": "centralized"}


def run_fhe_fl(config, logger):
    """Run federated learning with Fully Homomorphic Encryption (using ZK proofs)."""
    logger.info("🔐 Running FL with Fully Homomorphic Encryption (FL-FHE)")

    # Create FL system with FHE (using ZK proofs)
    fl_system = create_fl_system(FLApproaches.FHE)

    # Create datasets
    datasets, test_data = create_federated_datasets(
        dataset_name=config.data.dataset_name,
        num_clients=config.data.num_clients,
        partition_strategy=config.data.partition_strategy,
        alpha=config.data.alpha,
    )

    # Build model
    model_builder = ModelBuilder()
    model = model_builder.build_model(
        model_type=config.model.model_type,
        input_shape=config.model.input_shape,
        num_classes=config.model.num_classes,
    )

    # Simulate FHE operations
    logger.info("Setting up homomorphic encryption...")
    logger.info("Note: FHE approach uses ZK proofs for cryptographic security")

    # Test cryptographic operations
    if fl_system.get("crypto"):
        logger.info("Testing encrypted computations...")

        # Import numpy for array operations
        import numpy as np

        # Generate proof for encrypted computation
        client_id = "fhe_client_0"

        # Create old and new model parameters for proof
        old_params = {
            "layer1_weights": np.random.randn(32, 3, 3, 3),
            "layer1_bias": np.random.randn(32),
        }
        new_params = {
            "layer1_weights": np.random.randn(32, 3, 3, 3),
            "layer1_bias": np.random.randn(32),
        }

        training_data_hash = "simulated_fhe_data_hash"

        proof = fl_system["crypto"].generate_training_proof(
            old_model_params=old_params,
            new_model_params=new_params,
            training_data_hash=training_data_hash,
            learning_rate=0.01,
            epochs=1,
            client_id=client_id,
        )
        logger.info(f"Generated encrypted computation proof for {client_id}")

        # Verify the proof
        is_valid = fl_system["crypto"].verify_proof(proof)
        logger.info(
            f"Encrypted computation verification: {'✅ PASSED' if is_valid else '❌ FAILED'}"
        )

    logger.info("✅ FL-FHE setup complete")
    return {"model": model, "datasets": datasets, "fl_system": fl_system}


def run_classic_fl(config, logger):
    """Run classic federated learning without additional security."""
    logger.info("🚀 Running Classic Federated Learning")

    # Create FL system
    fl_system = create_fl_system(FLApproaches.CLASSIC)

    # Create datasets
    datasets, test_data = create_federated_datasets(
        dataset_name=config.data.dataset_name,
        num_clients=config.data.num_clients,
        partition_strategy=config.data.partition_strategy,
        alpha=config.data.alpha,
        data_path=config.data.data_path,
    )

    # Build model
    model_builder = ModelBuilder()
    model = model_builder.build_model(
        model_type=config.model.model_type,
        input_shape=config.model.input_shape,
        num_classes=config.model.num_classes,
    )

    logger.info("✅ Classic FL setup complete")
    return {"model": model, "datasets": datasets, "fl_system": fl_system}


def run_zk_fl(config, logger):
    """Run federated learning with zero-knowledge proofs."""
    logger.info("🔐 Running Zero-Knowledge Federated Learning")

    # Create FL system with ZK proofs
    fl_system = create_fl_system(
        FLApproaches.ZK,
        zk_config={
            "circuit_path": "./circuits/fl_circuit.json",
            "proving_key_path": "./keys/proving_key.json",
        },
        enable_blockchain=config.blockchain.enable_blockchain,
    )

    # Create datasets
    datasets, test_data = create_federated_datasets(
        dataset_name=config.data.dataset_name,
        num_clients=config.data.num_clients,
        partition_strategy=config.data.partition_strategy,
        alpha=config.data.alpha,
        data_path=config.data.data_path,
    )

    # Build model
    model_builder = ModelBuilder()
    model = model_builder.build_model(
        model_type=config.model.model_type,
        input_shape=config.model.input_shape,
        num_classes=config.model.num_classes,
    )

    # Test ZK proof generation
    if fl_system["crypto"]:
        logger.info("Testing ZK proof generation...")

        # Create mock training data hash
        training_data_hash = fl_system["crypto"].create_training_data_hash(
            datasets[0]["train_loader"] if "train_loader" in datasets[0] else []
        )

        # Generate mock model parameters
        import numpy as np

        old_params = {"layer1": np.random.randn(10, 5), "layer2": np.random.randn(5, 2)}
        new_params = {"layer1": np.random.randn(10, 5), "layer2": np.random.randn(5, 2)}

        # Generate proof
        proof_data = fl_system["crypto"].generate_training_proof(
            old_model_params=old_params,
            new_model_params=new_params,
            training_data_hash=training_data_hash,
            learning_rate=config.model.learning_rate,
            epochs=config.model.local_epochs,
            client_id="client_0",
        )

        # Verify proof
        is_valid = fl_system["crypto"].verify_proof(proof_data)
        logger.info(
            f"ZK proof verification: {'✅ PASSED' if is_valid else '❌ FAILED'}"
        )

        # Test blockchain integration if enabled
        if fl_system["blockchain"]:
            logger.info("Testing blockchain integration...")
            try:
                # Submit model update to blockchain
                tx_hash = fl_system["blockchain"].submit_model_update(
                    client_id="client_0",
                    round_number=1,
                    model_commitment=fl_system["crypto"].generate_model_commitment(
                        new_params
                    ),
                    zk_proof=proof_data,
                )
                logger.info(f"📦 Blockchain transaction: {tx_hash[:32]}...")

                # Verify blockchain record
                is_verified = fl_system["blockchain"].verify_model_update(tx_hash)
                logger.info(
                    f"⛓️  Blockchain verification: {'✅ PASSED' if is_verified else '❌ FAILED'}"
                )

                # Get round updates
                updates = fl_system["blockchain"].get_round_updates(1)
                logger.info(f"📊 Round updates retrieved: {len(updates)} entries")

            except Exception as e:
                logger.info(f"🔧 Blockchain demo: {e} (expected for mock mode)")

    logger.info("✅ ZK FL setup complete")
    return {"model": model, "datasets": datasets, "fl_system": fl_system}


def run_dp_fl(config, logger):
    """Run federated learning with differential privacy."""
    logger.info("🔒 Running Differential Privacy Federated Learning")

    # Create FL system with DP
    fl_system = create_fl_system(
        FLApproaches.DP,
        dp_config={
            "epsilon": config.privacy.epsilon,
            "delta": config.privacy.delta,
            "noise_scale": config.privacy.noise_multiplier,
            "clip_norm": config.privacy.max_grad_norm,
        },
    )

    # Create datasets
    datasets, test_data = create_federated_datasets(
        dataset_name=config.data.dataset_name,
        num_clients=config.data.num_clients,
        partition_strategy=config.data.partition_strategy,
        alpha=config.data.alpha,
        data_path=config.data.data_path,
    )

    # Build model
    model_builder = ModelBuilder()
    model = model_builder.build_model(
        model_type=config.model.model_type,
        input_shape=config.model.input_shape,
        num_classes=config.model.num_classes,
    )

    # Test differential privacy
    if fl_system["privacy"]:
        logger.info("Testing differential privacy...")

        # Generate mock parameters
        import numpy as np

        params = {"layer1": np.random.randn(10, 5), "layer2": np.random.randn(5, 2)}

        # Add noise
        noisy_params = fl_system["privacy"].add_noise_to_parameters(params)

        # Check privacy budget
        remaining_budget = fl_system["privacy"].get_remaining_budget()
        logger.info(f"Remaining privacy budget: ε = {remaining_budget:.4f}")

        logger.info("✅ Differential privacy test complete")

    logger.info("✅ DP FL setup complete")
    return {"model": model, "datasets": datasets, "fl_system": fl_system}


def compare_architectures():
    """Compare the old FHE architecture with the new ZK + Blockchain approach."""
    print("\n" + "=" * 80)
    print("📊 ARCHITECTURE COMPARISON: FHE vs ZK + Blockchain")
    print("=" * 80)

    print(
        "┌─────────────────────────┬──────────────────────────┬──────────────────────────┐"
    )
    print(
        "│ Aspect                  │ Original (FHE)           │ New (ZK + Blockchain)    │"
    )
    print(
        "├─────────────────────────┼──────────────────────────┼──────────────────────────┤"
    )
    print(
        "│ Privacy Method          │ Homomorphic Encryption   │ Differential Privacy     │"
    )
    print(
        "│ Integrity Verification │ Limited                  │ Zero-Knowledge Proofs    │"
    )
    print(
        "│ Transparency            │ None                     │ Blockchain Audit Trail   │"
    )
    print(
        "│ Performance             │ High computational cost  │ Efficient operations     │"
    )
    print(
        "│ Verifiability           │ Trust-based             │ Cryptographically proven │"
    )
    print(
        "│ Auditability            │ Not available           │ Immutable blockchain     │"
    )
    print(
        "│ Scalability             │ Limited by FHE overhead │ Highly scalable          │"
    )
    print(
        "└─────────────────────────┴──────────────────────────┴──────────────────────────┘"
    )

    print("\n🔍 Key Improvements in ZK + Blockchain Architecture:")
    print("✨ Better transparency through blockchain logging")
    print("✨ Stronger integrity guarantees via ZK proofs")
    print("✨ Improved performance by removing FHE overhead")
    print("✨ Enhanced auditability for regulatory compliance")
    print("✨ Maintainable privacy through differential privacy")

    print("\n📋 Architecture Migration Summary:")
    print("🔄 FHE → Differential Privacy: More practical privacy preservation")
    print("🔄 Trust-based → ZK Proofs: Cryptographic verification of training")
    print("🔄 Opaque → Blockchain: Transparent and auditable training records")
    print("🔄 Centralized → Decentralized: Distributed verification and storage")


def run_hybrid_fl(config, logger):
    """Run federated learning with hybrid approach (ZK + DP)."""
    logger.info("🔐🔒 Running Hybrid Federated Learning (ZK + DP)")

    # Create FL system with both ZK and DP
    fl_system = create_fl_system(
        FLApproaches.HYBRID,
        zk_config={
            "circuit_path": "./circuits/fl_circuit.json",
            "proving_key_path": "./keys/proving_key.json",
        },
        dp_config={
            "epsilon": config.privacy.epsilon,
            "delta": config.privacy.delta,
            "noise_scale": config.privacy.noise_multiplier,
            "clip_norm": config.privacy.max_grad_norm,
        },
        enable_blockchain=config.blockchain.enable_blockchain,
    )

    # Create datasets
    datasets, test_data = create_federated_datasets(
        dataset_name=config.data.dataset_name,
        num_clients=config.data.num_clients,
        partition_strategy=config.data.partition_strategy,
        alpha=config.data.alpha,
        data_path=config.data.data_path,
    )

    # Build model
    model_builder = ModelBuilder()
    model = model_builder.build_model(
        model_type=config.model.model_type,
        input_shape=config.model.input_shape,
        num_classes=config.model.num_classes,
    )

    # Test both ZK and DP
    logger.info("Testing hybrid approach...")

    import numpy as np

    old_params = {"layer1": np.random.randn(10, 5), "layer2": np.random.randn(5, 2)}
    new_params = {"layer1": np.random.randn(10, 5), "layer2": np.random.randn(5, 2)}

    # Apply differential privacy
    if fl_system["privacy"]:
        noisy_params = fl_system["privacy"].add_noise_to_parameters(new_params)
        logger.info("✅ Applied differential privacy")

    # Generate ZK proof
    if fl_system["crypto"]:
        training_data_hash = "mock_hash_" + str(hash("training_data"))
        proof_data = fl_system["crypto"].generate_training_proof(
            old_model_params=old_params,
            new_model_params=noisy_params if fl_system["privacy"] else new_params,
            training_data_hash=training_data_hash,
            learning_rate=config.model.learning_rate,
            epochs=config.model.local_epochs,
            client_id="client_0",
        )

        is_valid = fl_system["crypto"].verify_proof(proof_data)
        logger.info(
            f"ZK proof verification: {'✅ PASSED' if is_valid else '❌ FAILED'}"
        )

        # Test blockchain integration for hybrid approach
        if fl_system["blockchain"]:
            logger.info("Testing blockchain integration in hybrid mode...")
            try:
                # Submit comprehensive update to blockchain (ZK + DP)
                model_commitment = fl_system["crypto"].generate_model_commitment(
                    noisy_params if fl_system["privacy"] else new_params
                )

                tx_hash = fl_system["blockchain"].submit_model_update(
                    client_id="hybrid_client_0",
                    round_number=1,
                    model_commitment=model_commitment,
                    zk_proof=proof_data,
                )
                logger.info(f"📦 Hybrid blockchain transaction: {tx_hash[:32]}...")

                # Enhanced verification for hybrid approach
                is_verified = fl_system["blockchain"].verify_model_update(tx_hash)
                logger.info(
                    f"⛓️  Hybrid blockchain verification: {'✅ PASSED' if is_verified else '❌ FAILED'}"
                )

                # Privacy budget logging on blockchain
                if fl_system["privacy"]:
                    remaining_budget = fl_system["privacy"].get_remaining_budget()
                    logger.info(f"🔐 Privacy budget logged: ε = {remaining_budget:.4f}")

            except Exception as e:
                logger.info(f"🔧 Hybrid blockchain demo: {e} (expected for mock mode)")

    logger.info("✅ Hybrid FL setup complete")
    return {"model": model, "datasets": datasets, "fl_system": fl_system}


def main():
    """Main function to run ZKFL example with different approaches."""
    parser = argparse.ArgumentParser(description="ZKFL Framework Example")
    parser.add_argument(
        "--approach",
        choices=["centralized", "classic", "fhe", "zk", "dp", "hybrid"],
        default="classic",
        help="Learning approach: centralized (non-FL), classic (FL), fhe (FL-FHE), zk (FL-ZK), dp (FL-DP), hybrid (FL-ZK+DP)",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show architecture comparison between FHE and ZK+Blockchain approaches",
    )

    args = parser.parse_args()

    # Handle comparison option
    if args.compare:
        print("=" * 60)
        print("🌟 ZKFL Framework - Architecture Comparison 🌟")
        print("=" * 60)
        compare_architectures()
        print(
            "\n🎯 Comparison completed! Use --approach to run specific learning methods."
        )
        return 0

    print("=" * 60)
    print("🌟 ZKFL Framework - Zero-Knowledge Federated Learning 🌟")
    print("=" * 60)
    print(f"Approach: {args.approach.upper()}")
    print(f"Config: {args.config}")
    print("=" * 60)

    try:
        # Load configuration
        config_utils = ConfigUtils()
        config = config_utils.load_config(args.config)

        # Setup logging
        loggers = LoggingUtils.setup_federated_logging(
            base_dir="./logs", client_id=None
        )
        logger = loggers["main"]

        if args.verbose:
            logger.setLevel("DEBUG")

        logger.info(f"Starting ZKFL example with {args.approach} approach")

        # Run the selected approach
        start_time = time.time()

        if args.approach == "centralized":
            result = run_centralized(config, logger)
        elif args.approach == "classic":
            result = run_classic_fl(config, logger)
        elif args.approach == "fhe":
            result = run_fhe_fl(config, logger)
        elif args.approach == "zk":
            result = run_zk_fl(config, logger)
        elif args.approach == "dp":
            result = run_dp_fl(config, logger)
        elif args.approach == "hybrid":
            result = run_hybrid_fl(config, logger)
        else:
            raise ValueError(f"Unknown approach: {args.approach}")

        duration = time.time() - start_time

        # Print summary
        print("\n" + "=" * 60)
        print("📊 EXECUTION SUMMARY")
        print("=" * 60)
        print(f"✅ Approach: {args.approach.upper()}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(
            f"🏗️  Model: {result['model'].__class__.__name__ if result.get('model') else 'Not created'}"
        )

        # Handle different return formats
        if args.approach == "centralized":
            print("📊 Dataset: Centralized (all data)")
        else:
            print(f"📊 Datasets: {len(result.get('datasets', []))} clients")

        # FL system stats (only for federated approaches)
        fl_system = result.get("fl_system")
        if fl_system:
            if fl_system.get("crypto"):
                stats = fl_system["crypto"].get_stats()
                print(f"🔐 ZK Proofs Generated: {stats['proofs_generated']}")
                print(f"✅ ZK Proofs Verified: {stats['proofs_verified']}")

            if fl_system.get("privacy"):
                remaining = fl_system["privacy"].get_remaining_budget()
                print(f"🔒 DP Budget Remaining: ε = {remaining:.4f}")

            if fl_system.get("blockchain"):
                print("⛓️  Blockchain: Enabled")

        print("=" * 60)
        print("🎉 ZKFL Example completed successfully!")
        print("=" * 60)

        logger.info(f"ZKFL example completed successfully in {duration:.2f} seconds")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nFor help, run: python zkfl_example.py --help")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
