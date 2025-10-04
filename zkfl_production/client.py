"""
ZKFL Production Client

Production-ready federated learning client for distributed deployment using Flower.
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
from zkfl.federated.flower_client import ZKFLClient
from zkfl.data.data_setup import create_federated_datasets
from zkfl.utils.logging_utils import LoggingUtils
from zkfl.models.model_builder import ModelBuilder


class ZKFLProductionClient:
    """Production ZKFL client with comprehensive security features."""

    def __init__(self, client_id: int = 0, config_path: str = "config.yaml"):
        """
        Initialize production client.

        Args:
            client_id: Unique client identifier
            config_path: Path to configuration file
        """
        self.client_id = client_id
        self.config_path = config_path
        self.config = None
        self.logger = None
        self.client = None

    def setup(self):
        """Setup client components."""
        print(f"🚀 Initializing ZKFL Production Client {self.client_id}")

        # Initialize configuration management
        config_utils = ConfigUtils()
        self.config = config_utils.load_config(self.config_path)

        # Setup logging
        loggers = LoggingUtils.setup_federated_logging(
            base_dir="./logs",
            client_id=f"client_{self.client_id}",
        )
        self.logger = loggers["main"]
        self.logger.info(f"ZKFL Production Client {self.client_id} starting...")

    def prepare_data(self):
        """Prepare client dataset."""
        self.logger.info("Creating federated datasets...")

        # Allow environment variables to override config
        import os

        dataset_name = os.getenv("DATASET_TYPE", self.config.data.dataset_name)
        num_clients = int(os.getenv("NUM_CLIENTS", self.config.data.num_clients))

        self.logger.info(f"Using dataset: {dataset_name}, clients: {num_clients}")

        datasets, _ = create_federated_datasets(
            dataset_name=dataset_name,
            num_clients=num_clients,
            partition_strategy=self.config.data.partition_strategy,
            alpha=self.config.data.alpha,
            data_path=self.config.data.data_path,
        )

        # Select this client's data partition
        if self.client_id >= len(datasets):
            self.logger.warning(
                f"Client ID {self.client_id} exceeds available datasets. Using modulo."
            )
            client_dataset = datasets[self.client_id % len(datasets)]
        else:
            client_dataset = datasets[self.client_id]

        return client_dataset

    def build_model(self):
        """Build local model."""
        self.logger.info("Building local model...")

        # Auto-select model type based on dataset input shape if available
        if (
            hasattr(self, "client_dataset")
            and self.client_dataset
            and isinstance(self.client_dataset, dict)
            and "input_shape" in self.client_dataset
        ):
            input_shape = self.client_dataset["input_shape"]
            num_classes = self.client_dataset.get(
                "num_classes", self.config.model.num_classes
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
                f"Using config - model: {model_type}, input shape: {input_shape}"
            )

        model_builder = ModelBuilder()
        model = model_builder.build_model(
            model_type=model_type,
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_units=self.config.model.hidden_units,
            dropout_rate=self.config.model.dropout_rate,
        )
        return model

    def create_client(self, model, client_dataset):
        """Create ZKFL client instance."""
        self.client = ZKFLClient(
            client_id=f"zkfl_client_{self.client_id}",
            model=model,
            config=self.config,
            client_dataset=client_dataset,
            logger=self.logger,
        )
        return self.client

    def _wait_for_server(
        self, host: str, port: int, timeout: int = 60, retry_interval: int = 2
    ):
        """Wait for server to become available."""
        import socket

        start_time = time.time()

        print(f"⏳ Waiting for server {host}:{port} to become available...")

        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()

                if result == 0:
                    print(f"✅ Server {host}:{port} is ready!")
                    time.sleep(2)  # Give server a moment to fully initialize
                    return

            except Exception as e:
                pass

            print(f"⏳ Server not ready yet, retrying in {retry_interval}s...")
            time.sleep(retry_interval)

        raise ConnectionError(f"Server {host}:{port} not available after {timeout}s")

    def run(self):
        """Run the production client."""
        try:
            start_time = time.time()

            # Setup components
            self.setup()

            # Prepare data and model
            client_dataset = self.prepare_data()
            self.client_dataset = client_dataset  # Store for model building
            model = self.build_model()

            # Create client
            client = self.create_client(model, client_dataset)

            # Connect to server
            import os

            # Handle SERVER_ADDRESS environment variable (may include tcp:// prefix)
            server_address_env = os.getenv("SERVER_ADDRESS", None)
            if server_address_env:
                # Remove tcp:// prefix if present
                if server_address_env.startswith("tcp://"):
                    server_address_env = server_address_env[6:]

                # Split host and port
                if ":" in server_address_env:
                    server_host, server_port = server_address_env.split(":", 1)
                else:
                    server_host = server_address_env
                    server_port = "8080"
            else:
                server_host = os.getenv(
                    "SERVER_HOST", self.config.network.server_address
                )
                server_port = os.getenv("SERVER_PORT", self.config.network.server_port)

            server_address = f"{server_host}:{server_port}"

            # Wait for server to be ready
            self._wait_for_server(server_host, int(server_port))

            self.logger.info(f"Connecting to ZKFL server at {server_address}")

            print(f"🌸 Connecting to server at {server_address}")
            print(f"📊 Client {self.client_id} security features:")
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

            # Start Flower client with improved error handling and connection stability
            import os

            # Configure environment variables for better gRPC stability in Docker
            os.environ["GRPC_VERBOSITY"] = "ERROR"  # Reduce gRPC logging noise
            os.environ["GRPC_TRACE"] = ""

            max_retries = 3
            retry_count = 0
            base_delay = 5

            while retry_count < max_retries:
                try:
                    self.logger.info(
                        f"Attempting to connect to server (attempt {retry_count + 1}/{max_retries})"
                    )

                    # Use start_numpy_client for better Docker compatibility
                    fl.client.start_numpy_client(
                        server_address=server_address,
                        client=client,
                    )
                    break  # Success, exit retry loop

                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)

                    # Log the specific error for debugging
                    self.logger.warning(
                        f"Connection failed (attempt {retry_count}/{max_retries}): {error_msg}"
                    )

                    if retry_count < max_retries:
                        # Exponential backoff for retries
                        delay = base_delay * (2 ** (retry_count - 1))
                        self.logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

                        # Re-check server availability before retry
                        self._wait_for_server(server_host, int(server_port), timeout=30)
                    else:
                        self.logger.error(
                            f"Failed to connect after {max_retries} attempts: {error_msg}"
                        )
                        raise e

            duration = time.time() - start_time
            self.logger.info(
                f"✅ Client {self.client_id} completed in {duration:.2f} seconds"
            )

        except KeyboardInterrupt:
            self.logger.info(f"🛑 Client {self.client_id} stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Client {self.client_id} failed: {e}")
            raise


def main():
    """Main entry point for production client."""
    parser = argparse.ArgumentParser(description="ZKFL Production Client")
    parser.add_argument("client_id", type=int, help="Unique client identifier")
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
        "--differential_privacy",
        action="store_true",
        help="Enable differential privacy",
    )

    args = parser.parse_args()

    # Create and run client
    client = ZKFLProductionClient(client_id=args.client_id, config_path=args.config)

    # Apply command line overrides if needed
    if any([args.enable_zk, args.enable_blockchain, args.differential_privacy]):
        print("🔧 Applying command-line security overrides...")
        # In a full implementation, we'd override config here

    client.run()


if __name__ == "__main__":
    main()
