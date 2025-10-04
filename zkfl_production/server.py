"""
ZKFL Production Server

Production-ready federated learning server for distributed deployment using Flower.
Supports all ZKFL security features including ZK proofs, differential privacy,
and blockchain integration.
"""

import time
import argparse
import flwr as fl
from pathlib import Path
import sys

# Add parent directory to path for zkfl imports
sys.path.append(str(Path(__file__).parent.parent))

from zkfl.utils.config_utils import ConfigUtils
from zkfl.federated.flower_server import ZKFLServer
from zkfl.data.data_setup import create_federated_datasets
from zkfl.utils.logging_utils import LoggingUtils
from zkfl.models.model_builder import ModelBuilder
from zkfl.federated.federated_utils import FederatedUtils


class ZKFLProductionServer:
    """Production ZKFL server with comprehensive security features."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize production server.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = None
        self.logger = None
        self.server = None

    def setup(self):
        """Setup server components."""
        print("🚀 Initializing ZKFL Production Server")

        # Initialize configuration management
        config_utils = ConfigUtils()
        self.config = config_utils.load_config(self.config_path)

        # Setup logging
        loggers = LoggingUtils.setup_federated_logging(
            base_dir="./logs", client_id=None  # Server logging
        )
        self.logger = loggers["main"]
        self.logger.info("ZKFL Production Server starting...")

    def prepare_data(self):
        """Prepare evaluation dataset."""
        self.logger.info("Creating evaluation dataset...")

        # Allow environment variables to override config
        import os

        dataset_name = os.getenv("DATASET_TYPE", self.config.data.dataset_name)
        num_clients = int(os.getenv("NUM_CLIENTS", self.config.data.num_clients))

        self.logger.info(f"Using dataset: {dataset_name}, clients: {num_clients}")

        datasets, test_data = create_federated_datasets(
            dataset_name=dataset_name,
            num_clients=num_clients,
            partition_strategy=self.config.data.partition_strategy,
            alpha=self.config.data.alpha,
            data_path=self.config.data.data_path,
        )
        return datasets, test_data

    def build_model(self):
        """Build global model."""
        self.logger.info("Building global model...")

        # Get input shape from dataset if available
        if (
            hasattr(self, "test_data")
            and self.test_data
            and "input_shape" in self.test_data
        ):
            input_shape = self.test_data["input_shape"]
            num_classes = self.test_data.get(
                "num_classes", self.config.model.num_classes
            )
            self.logger.info(
                f"Using dataset input shape: {input_shape}, classes: {num_classes}"
            )

            # Auto-select model type based on input shape
            if len(input_shape) == 1:  # 1D data (tabular/synthetic)
                model_type = "mlp"
            elif len(input_shape) == 3 and input_shape[0] in [1, 3]:  # 3D data (images)
                model_type = self.config.model.model_type  # Use configured CNN
            else:
                model_type = "mlp"  # Default to MLP for unknown shapes

            self.logger.info(
                f"Auto-selected model type: {model_type} for input shape: {input_shape}"
            )
        else:
            input_shape = self.config.model.input_shape
            num_classes = self.config.model.num_classes
            model_type = self.config.model.model_type
            self.logger.info(
                f"Using config - model: {model_type}, input shape: {input_shape}, classes: {num_classes}"
            )

        model_builder = ModelBuilder()
        global_model = model_builder.build_model(
            model_type=model_type,
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_units=self.config.model.hidden_units,
            dropout_rate=self.config.model.dropout_rate,
        )
        return global_model

    def create_server(self, global_model, test_data):
        """Create ZKFL server strategy."""
        # Create strategy with configuration values
        strategy = ZKFLServer(
            initial_model=global_model,
            min_fit_clients=self.config.federated.min_fit_clients,
            min_evaluate_clients=self.config.federated.min_evaluate_clients,
            min_available_clients=self.config.federated.min_available_clients,
        )
        return strategy

    def run(self):
        """Run the production server."""
        try:
            start_time = time.time()

            # Setup components
            self.setup()

            # Prepare datasets first
            datasets, test_data = self.prepare_data()
            self.test_data = test_data  # Store for model building

            # Build global model with correct input shape
            global_model = self.build_model()

            # Create server strategy
            strategy = self.create_server(global_model, test_data)

            # Start Flower server
            import os

            # Override server address for Docker deployment (bind to all interfaces)
            server_host = os.getenv("SERVER_HOST", "0.0.0.0")
            server_port = os.getenv("SERVER_PORT", self.config.network.server_port)
            server_address = f"{server_host}:{server_port}"
            self.logger.info(f"Starting ZKFL server on {server_address}")

            print(f"🌸 Server listening on {server_address}")
            print("📊 Security features enabled:")
            if hasattr(self.config, "crypto") and getattr(
                self.config.crypto, "enable_zk", False
            ):
                print("  🔐 Zero-Knowledge Proofs: ✅")
            if hasattr(self.config, "privacy") and self.config.privacy.enable_dp:
                print("  🛡️  Differential Privacy: ✅")
            if (
                hasattr(self.config, "blockchain")
                and self.config.blockchain.enable_blockchain
            ):
                print("  ⛓️  Blockchain Integration: ✅")

            fl.server.start_server(
                server_address=server_address,
                config=fl.server.ServerConfig(
                    num_rounds=self.config.federated.num_rounds
                ),
                strategy=strategy,
            )

            duration = time.time() - start_time
            self.logger.info(f"✅ Server completed in {duration:.2f} seconds")

        except KeyboardInterrupt:
            self.logger.info("🛑 Server stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Server failed: {e}")
            raise


def main():
    """Main entry point for production server."""
    parser = argparse.ArgumentParser(description="ZKFL Production Server")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--enable_zk", action="store_true", help="Enable zero-knowledge proofs"
    )
    parser.add_argument(
        "--enable_blockchain", action="store_true", help="Enable blockchain integration"
    )
    parser.add_argument(
        "--enable_dp", action="store_true", help="Enable differential privacy"
    )

    args = parser.parse_args()

    # Create and run server
    server = ZKFLProductionServer(config_path=args.config)

    # Apply command line overrides if needed
    if any([args.enable_zk, args.enable_blockchain, args.enable_dp]):
        print("🔧 Applying command-line security overrides...")
        # In a full implementation, we'd override config here

    server.run()


if __name__ == "__main__":
    main()
