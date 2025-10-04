"""
Configuration management for the ZKFL framework.

This module provides centralized configuration management with support for
environment variables, YAML files, and programmatic configuration.
"""

import os
import yaml
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field

from .exceptions import ConfigurationError


@dataclass
class ZKConfig:
    """Configuration for zero-knowledge proof system."""

    circuit_path: str = "./circuits/fl_circuit.json"
    proving_key_path: str = "./keys/proving_key.json"
    verification_key_path: str = "./keys/verification_key.json"
    proof_save_path: str = "./proofs"
    enabled: bool = True


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy."""

    enabled: bool = True
    noise_scale: float = 1.0
    clip_norm: float = 1.0
    epsilon: float = 1.0
    delta: float = 1e-5
    secure_aggregation: bool = False


@dataclass
class BlockchainConfig:
    """Configuration for blockchain integration."""

    enabled: bool = False
    provider_url: str = "http://localhost:8545"
    contract_address: Optional[str] = None
    private_key: Optional[str] = None
    gas_limit: int = 3000000
    gas_price: Optional[int] = None


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""

    num_rounds: int = 3
    num_clients: int = 3
    min_fit_clients: int = 2
    min_eval_clients: int = 2
    min_available_clients: int = 2
    fraction_fit: float = 1.0
    fraction_eval: float = 1.0
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.001


@dataclass
class ModelConfig:
    """Configuration for model settings."""

    model_type: str = "cnn"
    num_classes: int = 10
    input_shape: tuple = (3, 32, 32)
    save_path: str = "./models"


@dataclass
class DataConfig:
    """Configuration for data settings."""

    dataset: str = "cifar10"
    data_path: str = "./data"
    batch_size: int = 64
    validation_split: float = 0.1
    num_workers: int = 4


@dataclass
class ZKFLConfig:
    """Main configuration class for ZKFL framework."""

    zk: ZKConfig = field(default_factory=ZKConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    blockchain: BlockchainConfig = field(default_factory=BlockchainConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # General settings
    device: str = "auto"
    seed: int = 42
    verbose: bool = True
    log_level: str = "INFO"


class ConfigManager:
    """Centralized configuration manager for ZKFL framework."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file (YAML)
        """
        self._config = ZKFLConfig()
        self._config_path = Path(config_path) if config_path else None

        if self._config_path and self._config_path.exists():
            self.load_from_file(self._config_path)

        # Override with environment variables
        self._load_from_environment()

    @property
    def config(self) -> ZKFLConfig:
        """Get the current configuration."""
        return self._config

    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)

            self._update_config_from_dict(config_dict)
            self._config_path = config_path

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")

    def save_to_file(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to YAML file.

        Args:
            config_path: Path to save configuration file. If None, uses current path.
        """
        if config_path:
            self._config_path = Path(config_path)

        if not self._config_path:
            raise ConfigurationError("No configuration path specified")

        # Ensure directory exists
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self._config_to_dict()

        try:
            with open(self._config_path, "w") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")

    def update(self, **kwargs) -> None:
        """
        Update configuration with keyword arguments.

        Args:
            **kwargs: Configuration updates using dot notation
                     e.g., zk_enabled=True, privacy_epsilon=2.0
        """
        self._update_config_from_dict(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'zk.enabled')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            obj = self._config
            for part in key.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'zk.enabled')
            value: Value to set
        """
        parts = key.split(".")
        obj = self._config

        # Navigate to parent object
        for part in parts[:-1]:
            obj = getattr(obj, part)

        # Set the final attribute
        setattr(obj, parts[-1], value)

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            "ZKFL_ZK_ENABLED": "zk.enabled",
            "ZKFL_PRIVACY_EPSILON": "privacy.epsilon",
            "ZKFL_BLOCKCHAIN_ENABLED": "blockchain.enabled",
            "ZKFL_BLOCKCHAIN_PROVIDER": "blockchain.provider_url",
            "ZKFL_NUM_ROUNDS": "federated.num_rounds",
            "ZKFL_NUM_CLIENTS": "federated.num_clients",
            "ZKFL_DEVICE": "device",
            "ZKFL_SEED": "seed",
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_key.endswith(".enabled"):
                    value = value.lower() in ("true", "1", "yes", "on")
                elif config_key in [
                    "privacy.epsilon",
                    "federated.fraction_fit",
                    "federated.fraction_eval",
                ]:
                    value = float(value)
                elif config_key in [
                    "federated.num_rounds",
                    "federated.num_clients",
                    "seed",
                ]:
                    value = int(value)

                self.set(config_key, value)

    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Handle nested dictionaries
                for nested_key, nested_value in value.items():
                    self.set(f"{key}.{nested_key}", nested_value)
            else:
                self.set(key, value)

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "zk": {
                "circuit_path": self._config.zk.circuit_path,
                "proving_key_path": self._config.zk.proving_key_path,
                "verification_key_path": self._config.zk.verification_key_path,
                "proof_save_path": self._config.zk.proof_save_path,
                "enabled": self._config.zk.enabled,
            },
            "privacy": {
                "enabled": self._config.privacy.enabled,
                "noise_scale": self._config.privacy.noise_scale,
                "clip_norm": self._config.privacy.clip_norm,
                "epsilon": self._config.privacy.epsilon,
                "delta": self._config.privacy.delta,
                "secure_aggregation": self._config.privacy.secure_aggregation,
            },
            "blockchain": {
                "enabled": self._config.blockchain.enabled,
                "provider_url": self._config.blockchain.provider_url,
                "contract_address": self._config.blockchain.contract_address,
                "private_key": self._config.blockchain.private_key,
                "gas_limit": self._config.blockchain.gas_limit,
                "gas_price": self._config.blockchain.gas_price,
            },
            "federated": {
                "num_rounds": self._config.federated.num_rounds,
                "num_clients": self._config.federated.num_clients,
                "min_fit_clients": self._config.federated.min_fit_clients,
                "min_eval_clients": self._config.federated.min_eval_clients,
                "min_available_clients": self._config.federated.min_available_clients,
                "fraction_fit": self._config.federated.fraction_fit,
                "fraction_eval": self._config.federated.fraction_eval,
                "local_epochs": self._config.federated.local_epochs,
                "batch_size": self._config.federated.batch_size,
                "learning_rate": self._config.federated.learning_rate,
            },
            "model": {
                "model_type": self._config.model.model_type,
                "num_classes": self._config.model.num_classes,
                "input_shape": self._config.model.input_shape,
                "save_path": self._config.model.save_path,
            },
            "data": {
                "dataset": self._config.data.dataset,
                "data_path": self._config.data.data_path,
                "batch_size": self._config.data.batch_size,
                "validation_split": self._config.data.validation_split,
                "num_workers": self._config.data.num_workers,
            },
            "device": self._config.device,
            "seed": self._config.seed,
            "verbose": self._config.verbose,
            "log_level": self._config.log_level,
        }

    def validate(self) -> List[str]:
        """
        Validate current configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate privacy settings
        if self._config.privacy.enabled:
            if self._config.privacy.epsilon <= 0:
                errors.append("Privacy epsilon must be positive")
            if self._config.privacy.delta < 0 or self._config.privacy.delta >= 1:
                errors.append("Privacy delta must be in [0, 1)")

        # Validate federated learning settings
        if self._config.federated.num_clients < 1:
            errors.append("Number of clients must be at least 1")
        if self._config.federated.min_fit_clients > self._config.federated.num_clients:
            errors.append("Min fit clients cannot exceed total clients")

        # Validate blockchain settings
        if self._config.blockchain.enabled:
            if not self._config.blockchain.provider_url:
                errors.append(
                    "Blockchain provider URL is required when blockchain is enabled"
                )

        return errors

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"ZKFLConfig(zk_enabled={self._config.zk.enabled}, privacy_enabled={self._config.privacy.enabled}, blockchain_enabled={self._config.blockchain.enabled})"
