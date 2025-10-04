"""
Docker-specific server wrapper for improved gRPC stability.
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
from zkfl.federated.flower_server import ZKFLServer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DockerServerWrapper:
    """Enhanced server wrapper for Docker environments with gRPC stability."""

    def __init__(self):
        self.shutdown_event = threading.Event()

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
            "GRPC_SO_REUSEADDR": "1",
        }

        for key, value in grpc_env.items():
            os.environ[key] = value

        logger.info(f"Configured gRPC environment for Docker: {grpc_env}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown_event.set()

    def start_server(self, server_address: str = "0.0.0.0:8080"):
        """Start the ZKFL server with enhanced Docker compatibility."""
        try:
            # Get configuration from environment
            min_fit_clients = int(os.getenv("MIN_FIT_CLIENTS", "1"))
            min_eval_clients = int(os.getenv("MIN_EVAL_CLIENTS", "1"))
            min_available_clients = int(os.getenv("MIN_AVAILABLE_CLIENTS", "1"))
            num_rounds = int(os.getenv("NUM_ROUNDS", "10"))

            logger.info(f"Starting ZKFL server on {server_address}")
            logger.info(
                f"Configuration: min_fit={min_fit_clients}, min_eval={min_eval_clients}, "
                f"min_available={min_available_clients}, rounds={num_rounds}"
            )

            # Create ZKFL server with robust configuration
            server = ZKFLServer()

            # Create strategy with Docker-optimized settings
            strategy = fl.server.strategy.FedAvg(
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_eval_clients,
                min_available_clients=min_available_clients,
                evaluate_fn=server.evaluate,  # Use server's evaluate function
                on_fit_config_fn=lambda server_round: {
                    "server_round": server_round,
                    "epochs": 1,
                    "batch_size": 32,
                    "learning_rate": 0.01,
                    "apply_privacy": True,
                    "generate_proof": True,
                },
                on_evaluate_config_fn=lambda server_round: {
                    "server_round": server_round,
                    "apply_privacy": False,
                    "generate_proof": False,
                },
            )

            # Configure server with enhanced error handling
            server_config = fl.server.ServerConfig(num_rounds=num_rounds)

            # Start server with robust error handling
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries and not self.shutdown_event.is_set():
                try:
                    retry_count += 1
                    logger.info(
                        f"Starting server (attempt {retry_count}/{max_retries})"
                    )

                    fl.server.start_server(
                        server_address=server_address,
                        config=server_config,
                        strategy=strategy,
                    )

                    # If we reach here, server completed successfully
                    logger.info("Server completed federated learning successfully")
                    return

                except Exception as e:
                    error_msg = str(e)
                    logger.warning(
                        f"Server start attempt {retry_count} failed: {error_msg}"
                    )

                    if retry_count < max_retries and not self.shutdown_event.is_set():
                        delay = 10 * retry_count  # Progressive delay
                        logger.info(f"Retrying server start in {delay} seconds...")

                        if self.shutdown_event.wait(timeout=delay):
                            logger.info("Shutdown requested during retry delay")
                            break
                    else:
                        logger.error(
                            f"Failed to start server after {max_retries} attempts"
                        )
                        raise e

            if self.shutdown_event.is_set():
                logger.info("Shutdown requested, stopping server")

        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server failed: {e}")
            raise


def main():
    """Main entry point for Docker server."""
    try:
        # Get configuration from environment
        server_host = os.getenv("SERVER_HOST", "0.0.0.0")
        server_port = os.getenv("SERVER_PORT", "8080")
        server_address = f"{server_host}:{server_port}"

        logger.info(f"Starting Docker ZKFL server on {server_address}")

        # Create and run server wrapper
        wrapper = DockerServerWrapper()
        wrapper.start_server(server_address)

        logger.info("Server completed successfully")

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
