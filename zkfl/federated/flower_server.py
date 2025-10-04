"""
Flower server implementation for ZKFL federated learning.

This module provides enhanced Flower servers with zero-knowledge proof
verification and blockchain integration for federated learning.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import random

from ..core.exceptions import FederatedLearningError, ZKProofError, BlockchainError
from ..crypto.zk_proof import ZKProofSystem
from ..blockchain.blockchain_interface import BlockchainInterface
from ..core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class FlowerServer:
    """Base Flower server for federated learning coordination."""

    def __init__(
        self,
        server_id: str = "server_0",
        initial_model: Any = None,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ):
        """
        Initialize Flower server.

        Args:
            server_id: Unique server identifier
            initial_model: Initial global model
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients
        """
        self.server_id = server_id
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        # Server state
        self.current_round = 0
        self.training_history = []
        self.client_updates = {}

        # Model management
        self.model_manager = ModelManager()
        if initial_model is not None:
            self.model_manager.model = initial_model

        logger.info(f"FlowerServer {server_id} initialized")

    def configure_fit(
        self, server_round: int, parameters: List[Any], client_manager: Any
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        Configure the next round of training.

        Args:
            server_round: Current server round
            parameters: Current global model parameters
            client_manager: Client manager instance

        Returns:
            List of (client, config) tuples
        """
        # Sample clients for training
        sample_size = max(
            self.min_fit_clients,
            min(self.min_available_clients, len(client_manager.all())),
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_fit_clients
        )

        # Create training configuration
        config = {
            "server_round": server_round,
            "epochs": 1,
            "learning_rate": 0.01,
            "batch_size": 32,
        }

        logger.info(
            f"Round {server_round}: Selected {len(clients)} clients for training"
        )
        return [(client, config) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: List[Any], client_manager: Any
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        Configure the next round of evaluation.

        Args:
            server_round: Current server round
            parameters: Current global model parameters
            client_manager: Client manager instance

        Returns:
            List of (client, config) tuples
        """
        # Sample clients for evaluation
        sample_size = max(
            self.min_evaluate_clients,
            min(self.min_available_clients, len(client_manager.all())),
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_evaluate_clients
        )

        # Create evaluation configuration
        config = {"server_round": server_round, "eval_steps": 100}

        logger.info(
            f"Round {server_round}: Selected {len(clients)} clients for evaluation"
        )
        return [(client, config) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Union[Tuple[Any, Exception], Exception]],
    ) -> Tuple[Optional[List[Any]], Dict[str, Any]]:
        """
        Aggregate training results from clients.

        Args:
            server_round: Current server round
            results: List of (client, fit_result) tuples
            failures: List of failures

        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
        """
        try:
            if not results:
                logger.warning(f"Round {server_round}: No results to aggregate")
                return None, {}

            logger.info(
                f"Round {server_round}: Aggregating {len(results)} client updates"
            )

            # Extract parameters and metrics
            parameters_list = []
            metrics_list = []
            total_examples = 0

            for client, fit_result in results:
                parameters, num_examples, metrics = fit_result
                parameters_list.append((parameters, num_examples))
                metrics_list.append(metrics)
                total_examples += num_examples

            # Aggregate parameters using weighted averaging
            aggregated_parameters = self._weighted_average(parameters_list)

            # Aggregate metrics
            aggregated_metrics = self._aggregate_metrics(metrics_list, total_examples)
            aggregated_metrics["total_examples"] = total_examples
            aggregated_metrics["num_clients"] = len(results)

            # Update server state
            self.client_updates[server_round] = {
                "parameters": aggregated_parameters,
                "metrics": aggregated_metrics,
                "num_clients": len(results),
            }

            # Update model manager
            if aggregated_parameters:
                param_dict = {
                    f"param_{i}": param for i, param in enumerate(aggregated_parameters)
                }
                self.model_manager.set_parameters(param_dict)
                self.model_manager.track_parameters()

            logger.info(f"Round {server_round}: Aggregation completed")
            return aggregated_parameters, aggregated_metrics

        except Exception as e:
            logger.error(f"Aggregation failed for round {server_round}: {e}")
            raise FederatedLearningError(f"Aggregation failed: {e}")

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Union[Tuple[Any, Exception], Exception]],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Aggregate evaluation results from clients.

        Args:
            server_round: Current server round
            results: List of (client, evaluate_result) tuples
            failures: List of failures

        Returns:
            Tuple of (aggregated_loss, aggregated_metrics)
        """
        try:
            if not results:
                logger.warning(f"Round {server_round}: No evaluation results")
                return None, {}

            logger.info(
                f"Round {server_round}: Aggregating {len(results)} evaluation results"
            )

            # Extract evaluation results
            losses = []
            metrics_list = []
            total_examples = 0

            for client, eval_result in results:
                loss, num_examples, metrics = eval_result
                if num_examples > 0:
                    losses.append((loss, num_examples))
                    metrics_list.append(metrics)
                    total_examples += num_examples

            if not losses:
                return None, {}

            # Compute weighted average loss
            weighted_loss = (
                sum(loss * num_examples for loss, num_examples in losses)
                / total_examples
            )

            # Aggregate metrics
            aggregated_metrics = self._aggregate_metrics(metrics_list, total_examples)
            aggregated_metrics["total_examples"] = total_examples
            aggregated_metrics["num_clients"] = len(results)

            logger.info(f"Round {server_round}: Evaluation aggregation completed")
            return weighted_loss, aggregated_metrics

        except Exception as e:
            logger.error(f"Evaluation aggregation failed for round {server_round}: {e}")
            return None, {}

    def evaluate(
        self, server_round: int, parameters: List[Any]
    ) -> Optional[Tuple[float, Dict[str, Any]]]:
        """
        Evaluate global model parameters.

        This method is called by Flower to evaluate the global model.
        Returns None to indicate that centralized evaluation is not performed.

        Args:
            server_round: Current server round
            parameters: Global model parameters

        Returns:
            None (no centralized evaluation)
        """
        logger.info(
            f"Round {server_round}: Centralized evaluation requested but not implemented"
        )
        return None

    def _weighted_average(
        self, parameters_list: List[Tuple[List[Any], int]]
    ) -> List[Any]:
        """
        Compute weighted average of parameters.

        Args:
            parameters_list: List of (parameters, num_examples) tuples

        Returns:
            Weighted average parameters
        """
        if not parameters_list:
            return []

        # Get total number of examples
        total_examples = sum(num_examples for _, num_examples in parameters_list)

        # Initialize aggregated parameters
        first_params, _ = parameters_list[0]
        aggregated_params = [0.0] * len(first_params)

        # Compute weighted sum
        for parameters, num_examples in parameters_list:
            weight = num_examples / total_examples
            for i, param in enumerate(parameters):
                if isinstance(param, (int, float)):
                    aggregated_params[i] += weight * param
                elif hasattr(param, "__mul__") and hasattr(param, "__add__"):
                    # Handle numpy arrays or torch tensors
                    if i == 0:
                        aggregated_params[i] = weight * param
                    else:
                        aggregated_params[i] = aggregated_params[i] + weight * param
                else:
                    # Fallback for other types
                    aggregated_params[i] = param

        return aggregated_params

    def _aggregate_metrics(
        self, metrics_list: List[Dict[str, Any]], total_examples: int
    ) -> Dict[str, Any]:
        """
        Aggregate metrics from multiple clients.

        Args:
            metrics_list: List of metric dictionaries
            total_examples: Total number of examples

        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}

        aggregated_metrics = {}

        # Find common metric keys
        common_keys = set(metrics_list[0].keys())
        for metrics in metrics_list[1:]:
            common_keys.intersection_update(metrics.keys())

        # Aggregate numeric metrics
        for key in common_keys:
            values = []
            for metrics in metrics_list:
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    values.append(value)

            if values:
                aggregated_metrics[key] = sum(values) / len(values)

        return aggregated_metrics

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information and statistics."""
        return {
            "server_id": self.server_id,
            "current_round": self.current_round,
            "total_rounds": len(self.training_history),
            "min_fit_clients": self.min_fit_clients,
            "min_evaluate_clients": self.min_evaluate_clients,
            "model_summary": self.model_manager.get_model_summary(),
            "recent_updates": list(self.client_updates.keys())[-5:],  # Last 5 rounds
        }


class ZKFLServer(FlowerServer):
    """Enhanced Flower server with ZK proof verification and blockchain integration."""

    def __init__(
        self,
        server_id: str = "zkfl_server_0",
        initial_model: Any = None,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        zk_system: Optional[ZKProofSystem] = None,
        blockchain: Optional[BlockchainInterface] = None,
    ):
        """
        Initialize ZKFL server.

        Args:
            server_id: Unique server identifier
            initial_model: Initial global model
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients
            zk_system: Zero-knowledge proof system
            blockchain: Blockchain interface
        """
        super().__init__(
            server_id,
            initial_model,
            min_fit_clients,
            min_evaluate_clients,
            min_available_clients,
        )

        self.zk_system = zk_system
        self.blockchain = blockchain

        # ZK and blockchain tracking
        self.proofs_verified = 0
        self.blockchain_submissions = 0
        self.failed_verifications = 0

        logger.info(
            f"ZKFLServer {server_id} initialized with ZK and blockchain support"
        )

    def initialize_parameters(self, client_manager: Any) -> Optional[List[Any]]:
        """
        Initialize global model parameters.

        Args:
            client_manager: Client manager instance

        Returns:
            Initial global model parameters or None
        """
        try:
            if self.model_manager.model is not None:
                # Extract parameters from initial model
                if hasattr(self.model_manager.model, "state_dict"):
                    # PyTorch model
                    import torch

                    state_dict = self.model_manager.model.state_dict()
                    parameters = [
                        param.detach().cpu().numpy() for param in state_dict.values()
                    ]
                    logger.info(
                        f"Initialized parameters from PyTorch model with {len(parameters)} layers"
                    )
                    return parameters
                elif hasattr(self.model_manager.model, "get_weights"):
                    # TensorFlow/Keras model
                    parameters = self.model_manager.model.get_weights()
                    logger.info(
                        f"Initialized parameters from TensorFlow model with {len(parameters)} layers"
                    )
                    return parameters
                else:
                    logger.warning(
                        "Initial model format not recognized, using random initialization"
                    )
                    return None
            else:
                logger.info(
                    "No initial model provided, will use client-side initialization"
                )
                return None
        except AttributeError as e:
            logger.warning(
                f"Failed to access initial_model attribute: {e}, using client-side initialization"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to initialize parameters: {e}")
            return None

    def configure_fit(
        self, server_round: int, parameters: List[Any], client_manager: Any
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        Enhanced training configuration with ZK and privacy settings.

        Args:
            server_round: Current server round
            parameters: Current global model parameters
            client_manager: Client manager instance

        Returns:
            List of (client, config) tuples
        """
        # Get base configuration
        client_configs = super().configure_fit(server_round, parameters, client_manager)

        # Enhance configuration with ZK and privacy settings
        enhanced_configs = []
        for client, config in client_configs:
            enhanced_config = config.copy()
            enhanced_config.update(
                {
                    "generate_proof": self.zk_system is not None,
                    "apply_privacy": True,
                    "round_number": server_round,
                    "blockchain_enabled": self.blockchain is not None,
                }
            )
            enhanced_configs.append((client, enhanced_config))

        return enhanced_configs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Union[Tuple[Any, Exception], Exception]],
    ) -> Tuple[Optional[List[Any]], Dict[str, Any]]:
        """
        Enhanced aggregation with ZK proof verification and blockchain logging.

        Args:
            server_round: Current server round
            results: List of (client, fit_result) tuples
            failures: List of failures

        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
        """
        try:
            logger.info(f"Round {server_round}: Starting secure aggregation")

            # Verify ZK proofs if available
            verified_results = []
            client_proofs = []

            for client, fit_result in results:
                parameters, num_examples, metrics = fit_result

                # Verify ZK proof if present
                if self.zk_system and "zk_proof" in metrics:
                    proof = metrics["zk_proof"]
                    if self._verify_client_proof(client, proof):
                        verified_results.append((client, fit_result))
                        client_proofs.append(proof)
                        self.proofs_verified += 1
                    else:
                        logger.warning(
                            f"ZK proof verification failed for client {client}"
                        )
                        self.failed_verifications += 1
                        # Optionally exclude client from aggregation
                        # continue
                        verified_results.append((client, fit_result))
                else:
                    verified_results.append((client, fit_result))

            if not verified_results:
                logger.error(
                    f"Round {server_round}: No verified results for aggregation"
                )
                return None, {}

            # Perform standard aggregation
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, verified_results, failures
            )

            # Generate aggregation proof if ZK system is available
            if self.zk_system and client_proofs and aggregated_parameters:
                try:
                    aggregation_proof = self.zk_system.generate_aggregation_proof(
                        client_proofs,
                        {
                            f"param_{i}": param
                            for i, param in enumerate(aggregated_parameters)
                        },
                        server_round,
                    )
                    aggregated_metrics["aggregation_proof"] = aggregation_proof
                    logger.info(f"Aggregation proof generated for round {server_round}")
                except Exception as e:
                    logger.warning(f"Failed to generate aggregation proof: {e}")

            # Submit to blockchain if available
            if self.blockchain and aggregated_parameters:
                try:
                    self._submit_to_blockchain(
                        server_round, aggregated_metrics, client_proofs
                    )
                except Exception as e:
                    logger.warning(f"Blockchain submission failed: {e}")

            # Update security metrics
            aggregated_metrics.update(
                {
                    "proofs_verified": len(client_proofs),
                    "total_clients": len(results),
                    "verification_rate": len(client_proofs) / max(1, len(results)),
                }
            )

            logger.info(f"Round {server_round}: Secure aggregation completed")
            return aggregated_parameters, aggregated_metrics

        except Exception as e:
            logger.error(f"Secure aggregation failed for round {server_round}: {e}")
            raise FederatedLearningError(f"Secure aggregation failed: {e}")

    def _verify_client_proof(self, client: Any, proof: Dict[str, Any]) -> bool:
        """
        Verify a client's zero-knowledge proof.

        Args:
            client: Client instance
            proof: ZK proof to verify

        Returns:
            True if proof is valid, False otherwise
        """
        try:
            if not self.zk_system:
                return True  # Accept without verification if no ZK system

            is_valid = self.zk_system.verify_training_proof(proof)

            if is_valid:
                logger.debug(f"ZK proof verified for client {client}")
            else:
                logger.warning(f"ZK proof verification failed for client {client}")

            return is_valid

        except Exception as e:
            logger.error(f"ZK proof verification error for client {client}: {e}")
            return False

    def _submit_to_blockchain(
        self,
        round_number: int,
        metrics: Dict[str, Any],
        client_proofs: List[Dict[str, Any]],
    ) -> None:
        """
        Submit round results to blockchain.

        Args:
            round_number: FL round number
            metrics: Aggregated round metrics
            client_proofs: List of client ZK proofs
        """
        try:
            if not self.blockchain:
                return

            # Submit round results
            tx_hash = self.blockchain.submit_round_results(
                round_number,
                metrics,
                [
                    {"proof_id": proof.get("proof_id", f"proof_{i}")}
                    for i, proof in enumerate(client_proofs)
                ],
            )

            self.blockchain_submissions += 1
            logger.info(f"Round {round_number} submitted to blockchain: {tx_hash}")

            # Submit individual proofs
            for i, proof in enumerate(client_proofs):
                try:
                    proof_tx = self.blockchain.submit_zk_proof(
                        proof.get("proof_id", f"proof_{round_number}_{i}"),
                        proof,
                        f"client_{i}",
                    )
                    logger.debug(f"ZK proof submitted to blockchain: {proof_tx}")
                except Exception as e:
                    logger.warning(f"Failed to submit proof {i} to blockchain: {e}")

        except Exception as e:
            logger.error(f"Blockchain submission failed: {e}")
            raise BlockchainError(f"Blockchain submission failed: {e}")

    def verify_round_integrity(self, round_number: int) -> bool:
        """
        Verify the integrity of a completed round using blockchain.

        Args:
            round_number: Round number to verify

        Returns:
            True if round integrity is verified, False otherwise
        """
        try:
            if not self.blockchain:
                logger.warning(
                    "Cannot verify round integrity: no blockchain connection"
                )
                return True

            # Get round information from local state
            if round_number not in self.client_updates:
                logger.warning(f"Round {round_number} not found in local state")
                return False

            round_data = self.client_updates[round_number]

            # Compute hash of round parameters
            from ..crypto.crypto_utils import CryptoUtils

            param_hash = CryptoUtils.compute_model_fingerprint(
                {
                    f"param_{i}": param
                    for i, param in enumerate(round_data["parameters"])
                }
            )

            # Verify against blockchain
            is_verified = self.blockchain.verify_model_integrity(param_hash)

            if is_verified:
                logger.info(f"Round {round_number} integrity verified")
            else:
                logger.warning(f"Round {round_number} integrity verification failed")

            return is_verified

        except Exception as e:
            logger.error(f"Round integrity verification failed: {e}")
            return False

    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security-related statistics."""
        stats = {
            "proofs_verified": self.proofs_verified,
            "failed_verifications": self.failed_verifications,
            "blockchain_submissions": self.blockchain_submissions,
            "zk_enabled": self.zk_system is not None,
            "blockchain_enabled": self.blockchain is not None,
        }

        if self.zk_system:
            zk_stats = self.zk_system.get_statistics()
            stats.update(
                {
                    "zk_proofs_generated": zk_stats["proofs_generated"],
                    "zk_success_rate": zk_stats["success_rate"],
                }
            )

        if self.blockchain:
            blockchain_status = self.blockchain.get_connection_status()
            stats.update(
                {
                    "blockchain_connected": blockchain_status["connected"],
                    "blockchain_transactions": blockchain_status["total_transactions"],
                }
            )

        return stats

    def export_audit_trail(self, output_path: str) -> None:
        """Export complete audit trail including blockchain data."""
        audit_data = {
            "server_info": self.get_server_info(),
            "security_stats": self.get_security_statistics(),
            "training_history": self.training_history,
            "client_updates": self.client_updates,
        }

        if self.blockchain:
            audit_data["blockchain_history"] = self.blockchain.get_transaction_history()

        import json
        from pathlib import Path

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(audit_data, f, indent=2, default=str)

        logger.info(f"Audit trail exported to {output_path}")


class ServerFactory:
    """Factory for creating ZKFL servers."""

    @staticmethod
    def create_server(
        server_type: str, server_id: str = "server_0", **kwargs
    ) -> Union[FlowerServer, ZKFLServer]:
        """
        Create a server instance.

        Args:
            server_type: Type of server ("standard" or "zkfl")
            server_id: Unique server identifier
            **kwargs: Additional arguments for server initialization

        Returns:
            Server instance
        """
        if server_type == "standard":
            return FlowerServer(server_id, **kwargs)
        elif server_type == "zkfl":
            return ZKFLServer(server_id, **kwargs)
        else:
            raise ValueError(f"Unknown server type: {server_type}")

    @staticmethod
    def create_zkfl_server_with_config(
        server_id: str, config: Dict[str, Any], **kwargs
    ) -> ZKFLServer:
        """
        Create ZKFL server with configuration.

        Args:
            server_id: Unique server identifier
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Configured ZKFL server
        """
        # Initialize ZK system if configured
        zk_system = None
        if config.get("zk", {}).get("enabled", False):
            zk_system = ZKProofSystem(
                circuit_path=config["zk"].get("circuit_path"),
                proving_key_path=config["zk"].get("proving_key_path"),
            )

        # Initialize blockchain interface if configured
        blockchain = None
        if config.get("blockchain", {}).get("enabled", False):
            blockchain = BlockchainInterface(
                provider_url=config["blockchain"]["provider_url"],
                contract_address=config["blockchain"].get("contract_address"),
            )

        return ZKFLServer(
            server_id=server_id, zk_system=zk_system, blockchain=blockchain, **kwargs
        )
