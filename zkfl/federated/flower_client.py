"""
Flower client implementation for ZKFL federated learning.

This module provides enhanced Flower clients with zero-knowledge proof
generation and differential privacy integration.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

try:
    from flwr.common import NDArrays, Parameters, Scalar
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.client import NumPyClient

    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    # Fallback types for when Flower is not available
    NDArrays = List[np.ndarray]
    Parameters = List[np.ndarray]
    Scalar = Union[bool, bytes, float, int, str]

    # Fallback NumPyClient base class
    class NumPyClient:
        def get_parameters(self, config):
            raise NotImplementedError

        def set_parameters(self, parameters):
            raise NotImplementedError

        def fit(self, parameters, config):
            raise NotImplementedError

        def evaluate(self, parameters, config):
            raise NotImplementedError


from ..core.exceptions import FederatedLearningError, ZKProofError, PrivacyError
from ..crypto.zk_proof import ZKProofSystem
from ..privacy.differential_privacy import DifferentialPrivacyManager
from ..core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class FlowerClient(NumPyClient):
    """Base Flower client for federated learning."""

    def __init__(
        self,
        client_id: str,
        model: Any = None,
        train_loader: Any = None,
        test_loader: Any = None,
    ):
        """
        Initialize Flower client.

        Args:
            client_id: Unique client identifier
            model: Local model for training
            train_loader: Training data loader
            test_loader: Test data loader (optional)
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Client state
        self.current_round = 0
        self.training_history = []

        # Model management
        self.model_manager = ModelManager()
        if model is not None:
            self.model_manager.model = model

        logger.info(f"FlowerClient {client_id} initialized")

    def get_parameters(self, config: Dict[str, Any] = None):
        """
        Get model parameters.

        Args:
            config: Configuration dictionary (from server)

        Returns:
            List of numpy arrays (required by NumPyClient)
        """
        try:
            parameters_dict = self.model_manager.get_parameters()
            logger.info(
                f"Client {self.client_id}: Retrieved {len(parameters_dict)} parameters"
            )

            # Convert to list of numpy arrays in consistent order
            parameter_arrays = list(parameters_dict.values())

            # Return list of numpy arrays as expected by NumPyClient
            return parameter_arrays

        except Exception as e:
            logger.error(f"Failed to get parameters: {e}")
            raise FederatedLearningError(f"Parameter retrieval failed: {e}")

    def set_parameters(self, parameters) -> None:
        """
        Set model parameters.

        Args:
            parameters: List of numpy arrays from server
        """
        try:
            # NumPyClient provides parameters as list of numpy arrays
            parameter_arrays = parameters

            # Convert list back to parameter dictionary
            param_dict = {}
            for i, param in enumerate(parameter_arrays):
                param_dict[f"param_{i}"] = param

            self.model_manager.set_parameters(param_dict)
            logger.info(
                f"Client {self.client_id}: Set {len(parameter_arrays)} parameters"
            )

        except Exception as e:
            logger.error(f"Failed to set parameters: {e}")
            raise FederatedLearningError(f"Parameter setting failed: {e}")

    def fit(
        self, parameters: List[Any], config: Dict[str, Any]
    ) -> Tuple[List[Any], int, Dict[str, Any]]:
        """
        Train model on local data.

        Args:
            parameters: Global model parameters from server
            config: Training configuration from server

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        try:
            logger.info(
                f"Client {self.client_id}: Starting training round {self.current_round + 1}"
            )

            # Set global parameters
            self.set_parameters(parameters)

            # Extract training configuration
            epochs = config.get("epochs", 1)
            learning_rate = config.get("learning_rate", 0.01)

            # Perform local training
            num_examples, metrics = self._train_local_model(epochs, learning_rate)

            # Get updated parameters
            updated_parameters = self.get_parameters()

            # Update training history
            self.training_history.append(
                {
                    "round": self.current_round + 1,
                    "num_examples": num_examples,
                    "metrics": metrics,
                    "config": config,
                }
            )

            self.current_round += 1

            logger.info(f"Client {self.client_id}: Training completed")
            return updated_parameters, num_examples, metrics

        except Exception as e:
            logger.error(f"Training failed for client {self.client_id}: {e}")
            raise FederatedLearningError(f"Local training failed: {e}")

    def evaluate(
        self, parameters: List[Any], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model on local test data.

        Args:
            parameters: Global model parameters
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        try:
            logger.info(f"Client {self.client_id}: Starting evaluation")

            # Set model parameters
            self.set_parameters(parameters)

            # Perform evaluation
            if self.test_loader is None:
                logger.warning("No test data available for evaluation")
                return 0.0, 0, {}

            loss, num_examples, metrics = self._evaluate_model()

            logger.info(f"Client {self.client_id}: Evaluation completed")
            return loss, num_examples, metrics

        except Exception as e:
            logger.error(f"Evaluation failed for client {self.client_id}: {e}")
            raise FederatedLearningError(f"Local evaluation failed: {e}")

    def _train_local_model(
        self, epochs: int, learning_rate: float
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Perform local model training.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for training

        Returns:
            Tuple of (num_examples, metrics)
        """
        # Simulated training process
        # In practice, this would implement actual PyTorch/TensorFlow training

        if self.train_loader is None:
            raise FederatedLearningError("No training data available")

        # Simulate training metrics
        num_examples = 1000  # Would be len(train_loader.dataset)

        # Mock training loop
        losses = []
        for epoch in range(epochs):
            epoch_loss = 2.5 - epoch * 0.1  # Simulated decreasing loss
            losses.append(epoch_loss)

        metrics = {
            "loss": losses[-1],
            "accuracy": 0.7 + (epochs * 0.05),  # Simulated improving accuracy
            "epochs": epochs,
            "learning_rate": learning_rate,
        }

        return num_examples, metrics

    def _evaluate_model(self) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model on test data.

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        if self.test_loader is None:
            return 0.0, 0, {}

        # Simulated evaluation
        num_examples = 200  # Would be len(test_loader.dataset)
        loss = 2.2  # Simulated test loss

        metrics = {
            "accuracy": 0.75,
            "precision": 0.73,
            "recall": 0.76,
            "f1_score": 0.74,
        }

        return loss, num_examples, metrics

    def get_client_info(self) -> Dict[str, Any]:
        """Get client information and statistics."""
        return {
            "client_id": self.client_id,
            "current_round": self.current_round,
            "total_rounds": len(self.training_history),
            "has_model": self.model is not None,
            "has_train_data": self.train_loader is not None,
            "has_test_data": self.test_loader is not None,
            "model_summary": self.model_manager.get_model_summary(),
        }


class ZKFLClient(FlowerClient):
    """Enhanced Flower client with ZK proofs and differential privacy."""

    def __init__(
        self,
        client_id: str = "zkfl_client_0",
        model: Any = None,
        train_loader: Any = None,
        test_loader: Any = None,
        zk_system: Optional[ZKProofSystem] = None,
        privacy_manager: Optional[DifferentialPrivacyManager] = None,
        config: Any = None,
        client_dataset: Any = None,
        logger: Any = None,
        **kwargs,
    ):
        """
        Initialize ZKFL client.

        Args:
            client_id: Unique client identifier
            model: Local model for training
            train_loader: Training data loader
            test_loader: Test data loader (optional)
            zk_system: Zero-knowledge proof system
            privacy_manager: Differential privacy manager
            config: Configuration object (for production compatibility)
            client_dataset: Client dataset (for production compatibility)
            logger: Logger instance (for production compatibility)
            **kwargs: Additional arguments for compatibility
        """
        # Handle train_loader from client_dataset if provided
        if client_dataset is not None and train_loader is None:
            train_loader = getattr(client_dataset, "train_loader", None)
        if client_dataset is not None and test_loader is None:
            test_loader = getattr(client_dataset, "test_loader", None)

        super().__init__(client_id, model, train_loader, test_loader)

        self.zk_system = zk_system
        self.privacy_manager = privacy_manager
        self.config = config
        self.client_dataset = client_dataset
        if logger:
            self.logger = logger

        # ZK and privacy tracking
        self.proofs_generated = 0
        self.privacy_budget_used = 0.0
        self._round_count = 0  # Track rounds for debugging
        self._connection_stable = True  # Track connection health

        logger.info(f"ZKFLClient {client_id} initialized with ZK and privacy support")

    def fit(
        self, parameters: List[Any], config: Dict[str, Any]
    ) -> Tuple[List[Any], int, Dict[str, Any]]:
        """
        Enhanced training with ZK proofs and differential privacy.

        Args:
            parameters: Global model parameters from server
            config: Training configuration from server

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        try:
            self._round_count += 1
            logger.info(
                f"ZKFLClient {self.client_id}: Starting secure training (round {self._round_count})"
            )

            # Store original parameters for ZK proof
            original_params = self.get_parameters()

            # Perform standard training
            updated_parameters, num_examples, metrics = super().fit(parameters, config)

            # Apply differential privacy if enabled
            if self.privacy_manager and config.get("apply_privacy", True):
                protected_params = self._apply_differential_privacy(updated_parameters)
                updated_parameters = protected_params
                metrics["privacy_applied"] = True

            # Generate ZK proof if enabled
            if self.zk_system and config.get("generate_proof", True):
                proof = self._generate_training_proof(
                    original_params, updated_parameters, num_examples
                )
                metrics["zk_proof"] = proof
                metrics["proof_generated"] = True
                self.proofs_generated += 1

            logger.info(f"ZKFLClient {self.client_id}: Secure training completed")
            return updated_parameters, num_examples, metrics

        except Exception as e:
            logger.error(f"Secure training failed for client {self.client_id}: {e}")
            raise FederatedLearningError(f"Secure training failed: {e}")

    def _apply_differential_privacy(self, parameters: List[Any]) -> List[Any]:
        """
        Apply differential privacy to model parameters.

        Args:
            parameters: Model parameters to protect

        Returns:
            Parameters with differential privacy applied
        """
        try:
            # Convert parameters to dictionary format
            param_dict = {}
            for i, param in enumerate(parameters):
                param_dict[f"param_{i}"] = param

            # Apply differential privacy
            protected_dict = self.privacy_manager.add_noise_to_parameters(param_dict)

            # Convert back to list format
            protected_parameters = [
                protected_dict[f"param_{i}"] for i in range(len(parameters))
            ]

            # Update privacy budget tracking
            stats = self.privacy_manager.get_statistics()
            self.privacy_budget_used = stats["privacy_budget_used"]

            logger.info(f"Differential privacy applied to {len(parameters)} parameters")
            return protected_parameters

        except Exception as e:
            logger.error(f"Failed to apply differential privacy: {e}")
            raise PrivacyError(f"Differential privacy application failed: {e}")

    def _generate_training_proof(
        self, original_params: List[Any], updated_params: List[Any], num_examples: int
    ) -> Dict[str, Any]:
        """
        Generate zero-knowledge proof for training.

        Args:
            original_params: Parameters before training
            updated_params: Parameters after training
            num_examples: Number of training examples

        Returns:
            Zero-knowledge proof
        """
        try:
            # Convert parameters to dictionary format for ZK system
            original_dict = {
                f"param_{i}": param for i, param in enumerate(original_params)
            }
            updated_dict = {
                f"param_{i}": param for i, param in enumerate(updated_params)
            }

            # Create training data hash (without revealing actual data)
            import hashlib

            training_data_hash = hashlib.sha256(
                f"{self.client_id}_{self.current_round}_{num_examples}".encode()
            ).hexdigest()

            # Generate ZK proof
            proof = self.zk_system.generate_training_proof(
                old_model_params=original_dict,
                new_model_params=updated_dict,
                training_data_hash=training_data_hash,
                client_id=self.client_id,
            )

            logger.info(f"ZK proof generated for client {self.client_id}")
            return proof

        except Exception as e:
            logger.error(f"Failed to generate ZK proof: {e}")
            raise ZKProofError(f"ZK proof generation failed: {e}")

    def verify_global_model(
        self, parameters: List[Any], proof: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Verify global model integrity using ZK proof.

        Args:
            parameters: Global model parameters to verify
            proof: Optional aggregation proof from server

        Returns:
            True if model is verified, False otherwise
        """
        try:
            if not self.zk_system or not proof:
                logger.warning("Cannot verify model: no ZK system or proof provided")
                return True  # Accept without verification

            # Verify the aggregation proof
            is_valid = self.zk_system.verify_aggregation_proof(proof)

            if is_valid:
                logger.info(
                    f"Global model verification successful for client {self.client_id}"
                )
            else:
                logger.warning(
                    f"Global model verification failed for client {self.client_id}"
                )

            return is_valid

        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False

    def get_privacy_budget_remaining(self) -> float:
        """Get remaining privacy budget."""
        if not self.privacy_manager:
            return float("inf")
        return self.privacy_manager.get_remaining_budget()

    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security and privacy statistics."""
        stats = {
            "proofs_generated": self.proofs_generated,
            "privacy_budget_used": self.privacy_budget_used,
            "zk_enabled": self.zk_system is not None,
            "privacy_enabled": self.privacy_manager is not None,
        }

        if self.privacy_manager:
            privacy_stats = self.privacy_manager.get_statistics()
            stats.update(
                {
                    "remaining_privacy_budget": privacy_stats["remaining_budget"],
                    "privacy_operations": privacy_stats["noise_operations"],
                }
            )

        if self.zk_system:
            zk_stats = self.zk_system.get_statistics()
            stats.update(
                {
                    "zk_proofs_verified": zk_stats["proofs_verified"],
                    "zk_success_rate": zk_stats["success_rate"],
                }
            )

        return stats

    def reset_security_state(self) -> None:
        """Reset security-related state."""
        self.proofs_generated = 0
        self.privacy_budget_used = 0.0

        if self.privacy_manager:
            self.privacy_manager.reset_budget()

        if self.zk_system:
            self.zk_system.clear_cache()

        logger.info(f"Security state reset for client {self.client_id}")

    def to_client(self):
        """
        Convert to Flower client format.

        This method is required by the Flower framework to properly
        integrate with start_client() function.

        Returns:
            Self (this client instance)
        """
        return self


class ClientFactory:
    """Factory for creating ZKFL clients."""

    @staticmethod
    def create_client(
        client_type: str, client_id: str, **kwargs
    ) -> Union[FlowerClient, ZKFLClient]:
        """
        Create a client instance.

        Args:
            client_type: Type of client ("standard" or "zkfl")
            client_id: Unique client identifier
            **kwargs: Additional arguments for client initialization

        Returns:
            Client instance
        """
        if client_type == "standard":
            return FlowerClient(client_id, **kwargs)
        elif client_type == "zkfl":
            return ZKFLClient(client_id, **kwargs)
        else:
            raise ValueError(f"Unknown client type: {client_type}")

    @staticmethod
    def create_zkfl_client_with_config(
        client_id: str, config: Dict[str, Any], **kwargs
    ) -> ZKFLClient:
        """
        Create ZKFL client with configuration.

        Args:
            client_id: Unique client identifier
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Configured ZKFL client
        """
        # Initialize ZK system if configured
        zk_system = None
        if config.get("zk", {}).get("enabled", False):
            zk_system = ZKProofSystem(
                circuit_path=config["zk"].get("circuit_path"),
                proving_key_path=config["zk"].get("proving_key_path"),
            )

        # Initialize privacy manager if configured
        privacy_manager = None
        if config.get("privacy", {}).get("enabled", False):
            privacy_manager = DifferentialPrivacyManager(
                noise_scale=config["privacy"].get("noise_scale", 1.0),
                clip_norm=config["privacy"].get("clip_norm", 1.0),
                epsilon=config["privacy"].get("epsilon", 1.0),
            )

        return ZKFLClient(
            client_id=client_id,
            zk_system=zk_system,
            privacy_manager=privacy_manager,
            **kwargs,
        )
