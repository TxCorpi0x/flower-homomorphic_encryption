"""
Core engine components for the ZKFL framework.

This module contains the main orchestration engines that coordinate
federated learning with zero-knowledge proofs and blockchain integration.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .config import ConfigManager, ZKFLConfig
from .model_manager import ModelManager
from .exceptions import FederatedLearningError, ConfigurationError

# Import other ZKFL components (will be created)
try:
    from ..crypto import ZKProofSystem
    from ..privacy import DifferentialPrivacyManager
    from ..blockchain import BlockchainInterface
    from ..federated import FlowerClient, FlowerServer

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrainingEngine:
    """Core training engine for local model training."""

    def __init__(self, config: ZKFLConfig):
        """
        Initialize training engine.

        Args:
            config: ZKFL configuration
        """
        self.config = config
        self.model_manager = ModelManager()
        self._setup_device()

    def _setup_device(self) -> None:
        """Setup computing device (CPU/GPU)."""
        try:
            import torch

            if self.config.device == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(self.config.device)

            logger.info(f"Using device: {self.device}")

        except ImportError:
            logger.warning("PyTorch not available, using CPU fallback")
            self.device = "cpu"

    def train_local_model(
        self, model: Any, train_loader: Any, epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train model locally for specified epochs.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            epochs: Number of epochs (uses config if None)

        Returns:
            Training metrics and updated parameters
        """
        if epochs is None:
            epochs = self.config.federated.local_epochs

        self.model_manager.model = model

        # Move model to device
        if hasattr(model, "to"):
            model.to(self.device)

        logger.info(f"Starting local training for {epochs} epochs")

        # Training loop implementation would go here
        # For now, return mock results
        return {
            "loss": 2.3,
            "accuracy": 0.1,
            "epochs_trained": epochs,
            "parameters": self.model_manager.get_parameters(),
        }

    def evaluate_model(self, model: Any, test_loader: Any) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            model: PyTorch model to evaluate
            test_loader: Test data loader

        Returns:
            Evaluation metrics
        """
        self.model_manager.model = model

        if hasattr(model, "to"):
            model.to(self.device)

        logger.info("Starting model evaluation")

        # Evaluation implementation would go here
        # For now, return mock results
        return {"loss": 2.25, "accuracy": 0.15}


class FederatedLearningEngine:
    """Main orchestration engine for ZKFL federated learning."""

    def __init__(
        self, config: Optional[ZKFLConfig] = None, config_path: Optional[str] = None
    ):
        """
        Initialize federated learning engine.

        Args:
            config: ZKFL configuration object
            config_path: Path to configuration file
        """
        # Initialize configuration
        if config is not None:
            self.config_manager = ConfigManager()
            self.config_manager._config = config
        else:
            self.config_manager = ConfigManager(config_path)

        self.config = self.config_manager.config

        # Validate configuration
        validation_errors = self.config_manager.validate()
        if validation_errors:
            raise ConfigurationError(
                f"Configuration validation failed: {validation_errors}"
            )

        # Initialize components
        self.training_engine = TrainingEngine(self.config)
        self.model_manager = ModelManager()

        # Initialize security components if dependencies are available
        if DEPENDENCIES_AVAILABLE:
            self._initialize_security_components()
        else:
            logger.warning("Some dependencies not available, running in limited mode")
            self.zk_system = None
            self.privacy_manager = None
            self.blockchain = None

        logger.info("FederatedLearningEngine initialized")

    def _initialize_security_components(self) -> None:
        """Initialize cryptographic and blockchain components."""
        try:
            # Initialize ZK proof system
            if self.config.zk.enabled:
                self.zk_system = ZKProofSystem(
                    circuit_path=self.config.zk.circuit_path,
                    proving_key_path=self.config.zk.proving_key_path,
                )
                logger.info("ZK proof system initialized")
            else:
                self.zk_system = None

            # Initialize differential privacy
            if self.config.privacy.enabled:
                self.privacy_manager = DifferentialPrivacyManager(
                    noise_scale=self.config.privacy.noise_scale,
                    clip_norm=self.config.privacy.clip_norm,
                    epsilon=self.config.privacy.epsilon,
                )
                logger.info("Differential privacy manager initialized")
            else:
                self.privacy_manager = None

            # Initialize blockchain interface
            if self.config.blockchain.enabled:
                self.blockchain = BlockchainInterface(
                    provider_url=self.config.blockchain.provider_url,
                    contract_address=self.config.blockchain.contract_address,
                )
                logger.info("Blockchain interface initialized")
            else:
                self.blockchain = None

        except Exception as e:
            logger.error(f"Failed to initialize security components: {e}")
            raise FederatedLearningError(
                f"Security component initialization failed: {e}"
            )

    def start_training(
        self, model: Any, train_datasets: List[Any], test_dataset: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Start federated learning training process.

        Args:
            model: Initial model to train
            train_datasets: List of training datasets for each client
            test_dataset: Test dataset for evaluation

        Returns:
            Training results and metrics
        """
        logger.info("Starting federated learning training")

        self.model_manager.model = model
        self.model_manager.track_parameters()

        results = {"rounds": [], "final_metrics": {}, "security_metrics": {}}

        # Federated learning rounds
        for round_num in range(1, self.config.federated.num_rounds + 1):
            logger.info(
                f"Starting round {round_num}/{self.config.federated.num_rounds}"
            )

            round_results = self._execute_round(
                round_num, model, train_datasets, test_dataset
            )

            results["rounds"].append(round_results)

        # Final evaluation
        if test_dataset:
            final_metrics = self.training_engine.evaluate_model(model, test_dataset)
            results["final_metrics"] = final_metrics

        # Security metrics
        if self.privacy_manager:
            results["security_metrics"][
                "remaining_privacy_budget"
            ] = self.privacy_manager.get_remaining_budget()

        if self.zk_system:
            results["security_metrics"]["total_proofs_generated"] = len(
                self.model_manager.get_integrity_hashes()
            )

        logger.info("Federated learning training completed")
        return results

    def _execute_round(
        self,
        round_num: int,
        model: Any,
        train_datasets: List[Any],
        test_dataset: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Execute a single federated learning round.

        Args:
            round_num: Current round number
            model: Current global model
            train_datasets: Training datasets for clients
            test_dataset: Test dataset

        Returns:
            Round results and metrics
        """
        # Get current model parameters
        global_params = self.model_manager.get_parameters()
        client_updates = []

        # Simulate client training
        num_selected_clients = min(
            self.config.federated.num_clients, len(train_datasets)
        )

        for client_id in range(num_selected_clients):
            logger.info(f"Training client {client_id}")

            # Train local model
            local_results = self.training_engine.train_local_model(
                model, train_datasets[client_id]
            )

            # Apply privacy protection
            if self.privacy_manager:
                protected_params = self.privacy_manager.add_noise_to_parameters(
                    local_results["parameters"]
                )
                local_results["parameters"] = protected_params

            # Generate ZK proof
            if self.zk_system:
                proof = self.zk_system.generate_training_proof(
                    old_model_params=global_params,
                    new_model_params=local_results["parameters"],
                    training_data_hash=f"client_{client_id}_round_{round_num}",
                    client_id=str(client_id),
                )
                local_results["zk_proof"] = proof

            client_updates.append(local_results)

        # Aggregate updates
        aggregated_params = self._aggregate_parameters(
            [update["parameters"] for update in client_updates]
        )

        # Update global model
        self.model_manager.set_parameters(aggregated_params)
        self.model_manager.track_parameters()

        # Evaluate updated model
        round_metrics = {}
        if test_dataset:
            round_metrics = self.training_engine.evaluate_model(model, test_dataset)

        # Log to blockchain
        if self.blockchain:
            try:
                self.blockchain.submit_round_results(
                    round_num, round_metrics, client_updates
                )
            except Exception as e:
                logger.warning(f"Failed to log to blockchain: {e}")

        return {
            "round": round_num,
            "metrics": round_metrics,
            "num_clients": len(client_updates),
            "parameter_hash": self.model_manager.compute_parameter_hash(),
        }

    def _aggregate_parameters(
        self, parameter_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate parameters from multiple clients.

        Args:
            parameter_list: List of parameter dictionaries from clients

        Returns:
            Aggregated parameters
        """
        if not parameter_list:
            raise FederatedLearningError("No parameters to aggregate")

        # Simple federated averaging
        aggregated = {}

        for param_name in parameter_list[0].keys():
            param_arrays = [params[param_name] for params in parameter_list]

            # Stack and take mean
            import numpy as np

            stacked = np.stack(param_arrays)
            aggregated[param_name] = np.mean(stacked, axis=0)

        return aggregated

    def save_checkpoint(self, path: str) -> None:
        """Save current training state."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = checkpoint_path / "model.pt"
        self.model_manager.save_model(model_path)

        # Save configuration
        config_path = checkpoint_path / "config.yaml"
        self.config_manager.save_to_file(config_path)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the federated learning engine."""
        status = {
            "config": {
                "zk_enabled": self.config.zk.enabled,
                "privacy_enabled": self.config.privacy.enabled,
                "blockchain_enabled": self.config.blockchain.enabled,
                "num_rounds": self.config.federated.num_rounds,
                "num_clients": self.config.federated.num_clients,
            },
            "model": self.model_manager.get_model_summary(),
            "components": {
                "zk_system": self.zk_system is not None,
                "privacy_manager": self.privacy_manager is not None,
                "blockchain": self.blockchain is not None,
            },
        }

        return status
