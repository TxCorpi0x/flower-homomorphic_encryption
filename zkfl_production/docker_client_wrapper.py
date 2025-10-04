"""
Docker-specific client wrapper for improved gRPC stability.
"""

import os
import sys
import time
import logging
import signal
import threading
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import flwr as fl
from zkfl.federated.flower_client import ZKFLClient
from zkfl.data.data_setup import DataSetup
from zkfl.models.model_builder import ModelBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DockerClientWrapper:
    """Enhanced client wrapper for Docker environments with gRPC stability."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.shutdown_event = threading.Event()
        self.connection_attempts = 0
        self.max_connection_attempts = 10

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Configure gRPC environment variables for Docker
        self._setup_grpc_environment()

    def _setup_grpc_environment(self):
        """Configure gRPC environment variables for better Docker compatibility."""
        grpc_env = {
            "GRPC_VERBOSITY": "ERROR",
            "GRPC_TRACE": "",
            "GRPC_KEEPALIVE_TIME_MS": "30000",
            "GRPC_KEEPALIVE_TIMEOUT_MS": "5000",
            "GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS": "1",
            "GRPC_HTTP2_MAX_PINGS_WITHOUT_DATA": "0",
            "GRPC_HTTP2_MIN_PING_INTERVAL_WITHOUT_DATA_MS": "300000",
            "GRPC_HTTP2_BDP_PROBE": "0",
            "GRPC_SO_REUSEPORT": "0",
        }

        for key, value in grpc_env.items():
            os.environ[key] = value

        logger.info(f"Configured gRPC environment for Docker: {grpc_env}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown_event.set()

    def _wait_for_server(self, host: str, port: int, timeout: int = 60):
        """Wait for server to be available with enhanced retry logic."""
        import socket

        end_time = time.time() + timeout
        attempt = 0

        while time.time() < end_time and not self.shutdown_event.is_set():
            attempt += 1
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5)
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        logger.info(
                            f"Server {host}:{port} is available (attempt {attempt})"
                        )
                        return True
            except Exception as e:
                logger.debug(f"Connection attempt {attempt} failed: {e}")

            if not self.shutdown_event.is_set():
                time.sleep(
                    min(2 ** min(attempt - 1, 4), 10)
                )  # Exponential backoff with cap

        raise TimeoutError(f"Server {host}:{port} not available after {timeout}s")

    def _create_robust_client(self) -> ZKFLClient:
        """Create a ZKFL client with enhanced error handling."""
        try:
            # Initialize data setup
            data_setup = DataSetup()

            # Create federated dataset for this client
            client_datasets, test_dataset = data_setup.create_federated_dataset(
                dataset_name="synthetic",
                num_clients=3,
                partition_method="iid",
                client_id=int(self.client_id),
                num_samples=1000,
                num_features=784,
                num_classes=10,
            )

            # Get this client's dataset
            client_dataset = client_datasets[
                int(self.client_id) - 1
            ]  # 1-indexed to 0-indexed

            # Initialize model builder
            model_builder = ModelBuilder()
            model = model_builder.build_model(
                model_type="mlp", input_shape=(784,), num_classes=10
            )

            # Create enhanced client with proper dataset structure
            client = ZKFLClient(
                client_id=self.client_id,
                model=model,
                train_loader=client_dataset,  # Use client dataset directly
                test_loader=test_dataset,
                client_dataset=client_dataset,
                logger=logger,  # Pass the logger explicitly
            )

            logger.info(f"Successfully created ZKFL client {self.client_id}")
            return client

        except Exception as e:
            logger.error(f"Failed to create client: {e}")
            raise

    def connect_with_retry(self, server_address: str):
        """Connect to server with enhanced retry logic and connection monitoring."""

        # Parse server address
        if ":" in server_address:
            host, port = server_address.split(":")
            port = int(port)
        else:
            host = server_address
            port = 8080

        logger.info(f"Connecting to server at {host}:{port}")

        # Wait for server availability
        self._wait_for_server(host, port, timeout=120)

        # Create client
        client = self._create_robust_client()

        # Connection loop with progressive backoff
        base_delay = 5
        max_delay = 60

        while (
            self.connection_attempts < self.max_connection_attempts
            and not self.shutdown_event.is_set()
        ):

            self.connection_attempts += 1
            delay = min(base_delay * (1.5 ** (self.connection_attempts - 1)), max_delay)

            try:
                logger.info(
                    f"Connection attempt {self.connection_attempts}/{self.max_connection_attempts}"
                )

                # Additional pre-connection checks
                self._wait_for_server(host, port, timeout=30)

                # Start Flower client with NumPy interface for better compatibility
                fl.client.start_numpy_client(
                    server_address=server_address,
                    client=client,
                )

                # If we reach here, connection was successful
                logger.info(
                    f"Client {self.client_id} completed federated learning successfully"
                )
                return

            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"Connection attempt {self.connection_attempts} failed: {error_msg}"
                )

                # Check for specific Docker/gRPC issues
                if "Socket closed" in error_msg or "UNAVAILABLE" in error_msg:
                    logger.info(
                        "Detected gRPC connection issue, applying extended recovery delay"
                    )
                    delay *= 2  # Double delay for gRPC issues

                if self.connection_attempts < self.max_connection_attempts:
                    logger.info(f"Retrying in {delay:.1f} seconds...")

                    # Use shutdown event for interruptible sleep
                    if self.shutdown_event.wait(timeout=delay):
                        logger.info("Shutdown requested during retry delay")
                        break
                else:
                    logger.error(
                        f"Failed to connect after {self.max_connection_attempts} attempts"
                    )
                    raise e

        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, stopping connection attempts")
        else:
            logger.error("Maximum connection attempts reached, giving up")


def main():
    """Main entry point for Docker client."""
    try:
        # Get configuration from environment
        client_id = os.getenv("CLIENT_ID", "1")
        server_host = os.getenv("SERVER_HOST", "zkfl-server")
        server_port = os.getenv("SERVER_PORT", "8080")
        server_address = f"{server_host}:{server_port}"

        logger.info(f"Starting Docker client {client_id} for server {server_address}")

        # Create and run client wrapper
        wrapper = DockerClientWrapper(client_id)
        wrapper.connect_with_retry(server_address)

        logger.info(f"Client {client_id} completed successfully")

    except KeyboardInterrupt:
        logger.info("Client interrupted by user")
    except Exception as e:
        logger.error(f"Client failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
