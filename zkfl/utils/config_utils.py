"""
Configuration utilities for ZKFL framework.

This module provides advanced configuration management with support for
environment-specific settings, validation, and dynamic configuration updates.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union, List, Type, get_type_hints
from pathlib import Path
from dataclasses import dataclass, field, fields
import logging

from ..core.exceptions import ConfigurationError


@dataclass
class NetworkConfig:
    """Network configuration for federated learning."""

    server_address: str = "localhost"
    server_port: int = 8080
    client_timeout: int = 30
    max_retries: int = 3
    use_ssl: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None


@dataclass
class PrivacyConfig:
    """Privacy configuration for differential privacy."""

    enable_dp: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    privacy_accountant: str = "rdp"  # "rdp" or "pld"


@dataclass
class CryptoConfig:
    """Cryptographic configuration."""

    enable_zk_proofs: bool = True
    proof_system: str = "groth16"  # "groth16", "plonk", "stark"
    key_size: int = 2048
    hash_algorithm: str = "sha256"
    encryption_algorithm: str = "aes256"
    enable_homomorphic: bool = False


@dataclass
class BlockchainConfig:
    """Blockchain configuration."""

    enable_blockchain: bool = False
    network_url: str = "http://localhost:8545"
    contract_address: Optional[str] = None
    private_key: Optional[str] = None
    gas_limit: int = 6000000
    gas_price: int = 20000000000  # 20 gwei


@dataclass
class ModelConfig:
    """Model configuration."""

    model_type: str = "cnn"
    input_shape: List[int] = field(default_factory=lambda: [32, 32, 3])
    num_classes: int = 10
    hidden_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.2
    activation: str = "relu"
    optimizer: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 32
    local_epochs: int = 5


@dataclass
class DataConfig:
    """Data configuration."""

    dataset_name: str = "cifar10"
    data_path: str = "./data"
    partition_strategy: str = "iid"  # "iid", "non_iid", "dirichlet"
    num_clients: int = 10
    alpha: float = 0.5  # For Dirichlet distribution
    min_samples_per_client: int = 100
    validation_split: float = 0.2
    normalize_data: bool = True
    augment_data: bool = False


@dataclass
class FederatedConfig:
    """Federated learning configuration."""

    num_rounds: int = 100
    fraction_fit: float = 0.1
    fraction_evaluate: float = 0.1
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    aggregation_strategy: str = "fedavg"  # "fedavg", "fedprox", "scaffold"
    server_round_timeout: Optional[int] = None


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_dir: str = "./logs"
    enable_structured_logging: bool = True
    enable_audit_logging: bool = True
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    experiment_name: str = "zkfl_experiment"
    output_dir: str = "./outputs"
    save_model: bool = True
    save_metrics: bool = True
    checkpoint_frequency: int = 10
    early_stopping_patience: int = 20
    random_seed: int = 42


@dataclass
class ZKFLConfig:
    """Main ZKFL configuration containing all sub-configurations."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    blockchain: BlockchainConfig = field(default_factory=BlockchainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


class ConfigUtils:
    """Advanced configuration management utilities."""

    _instance: Optional["ConfigUtils"] = None
    _config: Optional[ZKFLConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._config_cache: Dict[str, Any] = {}
            self._validation_rules: Dict[str, List[callable]] = {}
            self._environment_overrides: Dict[str, str] = {}
            self._initialized = True

    @classmethod
    def get_instance(cls) -> "ConfigUtils":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_config(
        self, config_path: Optional[str] = None, environment: Optional[str] = None
    ) -> ZKFLConfig:
        """
        Load configuration from file with environment overrides.

        Args:
            config_path: Path to configuration file
            environment: Environment name (dev, prod, test)

        Returns:
            Loaded configuration
        """
        # Start with default configuration
        config = ZKFLConfig()

        # Load from file if provided
        if config_path:
            config = self._load_config_from_file(config_path, config)

        # Apply environment-specific overrides
        if environment:
            config = self._apply_environment_overrides(config, environment)

        # Apply environment variable overrides
        config = self._apply_env_var_overrides(config)

        # Validate configuration
        self._validate_config(config)

        # Cache the configuration
        self._config = config

        return config

    def _load_config_from_file(
        self, config_path: str, base_config: ZKFLConfig
    ) -> ZKFLConfig:
        """Load configuration from YAML or JSON file."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, "r") as f:
                if config_file.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                elif config_file.suffix.lower() == ".json":
                    config_data = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {config_file.suffix}"
                    )

            # Update configuration with loaded data
            return self._update_config_from_dict(base_config, config_data)

        except Exception as e:
            raise ConfigurationError(f"Error loading configuration file: {e}")

    def _update_config_from_dict(
        self, config: ZKFLConfig, config_dict: Dict[str, Any]
    ) -> ZKFLConfig:
        """Update configuration object from dictionary."""
        for section_name, section_data in config_dict.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section_config = getattr(config, section_name)

                for key, value in section_data.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                    else:
                        logging.warning(
                            f"Unknown configuration key: {section_name}.{key}"
                        )

        return config

    def _apply_environment_overrides(
        self, config: ZKFLConfig, environment: str
    ) -> ZKFLConfig:
        """Apply environment-specific configuration overrides."""
        env_config_path = f"config_{environment}.yaml"

        if Path(env_config_path).exists():
            config = self._load_config_from_file(env_config_path, config)

        return config

    def _apply_env_var_overrides(self, config: ZKFLConfig) -> ZKFLConfig:
        """Apply environment variable overrides."""
        # Define mapping from environment variables to config paths
        env_mappings = {
            "ZKFL_SERVER_ADDRESS": ("network", "server_address"),
            "ZKFL_SERVER_PORT": ("network", "server_port"),
            "ZKFL_EPSILON": ("privacy", "epsilon"),
            "ZKFL_DELTA": ("privacy", "delta"),
            "ZKFL_NUM_ROUNDS": ("federated", "num_rounds"),
            "ZKFL_NUM_CLIENTS": ("data", "num_clients"),
            "ZKFL_LEARNING_RATE": ("model", "learning_rate"),
            "ZKFL_BATCH_SIZE": ("model", "batch_size"),
            "ZKFL_LOG_LEVEL": ("logging", "level"),
            "ZKFL_EXPERIMENT_NAME": ("experiment", "experiment_name"),
        }

        for env_var, (section, key) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                section_config = getattr(config, section)

                # Convert to appropriate type
                field_type = type(getattr(section_config, key))
                try:
                    if field_type == bool:
                        converted_value = env_value.lower() in (
                            "true",
                            "1",
                            "yes",
                            "on",
                        )
                    elif field_type == int:
                        converted_value = int(env_value)
                    elif field_type == float:
                        converted_value = float(env_value)
                    else:
                        converted_value = env_value

                    setattr(section_config, key, converted_value)
                except ValueError as e:
                    logging.warning(f"Invalid value for {env_var}: {env_value} ({e})")

        return config

    def _validate_config(self, config: ZKFLConfig):
        """Validate configuration values."""
        errors = []

        # Network validation
        if config.network.server_port <= 0 or config.network.server_port > 65535:
            errors.append("Server port must be between 1 and 65535")

        # Privacy validation
        if config.privacy.enable_dp:
            if config.privacy.epsilon <= 0:
                errors.append("Privacy epsilon must be positive")
            if config.privacy.delta < 0 or config.privacy.delta >= 1:
                errors.append("Privacy delta must be between 0 and 1")

        # Model validation
        if config.model.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        if config.model.batch_size <= 0:
            errors.append("Batch size must be positive")
        if config.model.local_epochs <= 0:
            errors.append("Local epochs must be positive")

        # Federated validation
        if config.federated.num_rounds <= 0:
            errors.append("Number of rounds must be positive")
        if config.federated.fraction_fit <= 0 or config.federated.fraction_fit > 1:
            errors.append("Fraction fit must be between 0 and 1")
        if config.federated.min_fit_clients > config.data.num_clients:
            errors.append("Min fit clients cannot exceed total number of clients")

        # Data validation
        if config.data.num_clients <= 0:
            errors.append("Number of clients must be positive")
        if config.data.validation_split < 0 or config.data.validation_split >= 1:
            errors.append("Validation split must be between 0 and 1")

        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )

    def save_config(
        self, config: ZKFLConfig, output_path: str, format_type: str = "yaml"
    ):
        """
        Save configuration to file.

        Args:
            config: Configuration to save
            output_path: Output file path
            format_type: Output format ("yaml" or "json")
        """
        config_dict = self._config_to_dict(config)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, "w") as f:
                if format_type.lower() == "yaml":
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format_type.lower() == "json":
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported format: {format_type}")

        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")

    def _config_to_dict(self, config: ZKFLConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        result = {}

        for field in fields(config):
            section_config = getattr(config, field.name)
            section_dict = {}

            for section_field in fields(section_config):
                value = getattr(section_config, section_field.name)
                section_dict[section_field.name] = value

            result[field.name] = section_dict

        return result

    def get_config(self) -> ZKFLConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates in format {section.key: value}
        """
        if self._config is None:
            self._config = ZKFLConfig()

        for key_path, value in updates.items():
            if "." not in key_path:
                continue

            section, key = key_path.split(".", 1)

            if hasattr(self._config, section):
                section_config = getattr(self._config, section)
                if hasattr(section_config, key):
                    setattr(section_config, key, value)

    def create_client_config(
        self, client_id: str, base_config: Optional[ZKFLConfig] = None
    ) -> ZKFLConfig:
        """
        Create client-specific configuration.

        Args:
            client_id: Client identifier
            base_config: Base configuration to copy from

        Returns:
            Client-specific configuration
        """
        if base_config is None:
            base_config = self.get_config()

        # Create a copy for the client
        import copy

        client_config = copy.deepcopy(base_config)

        # Apply client-specific modifications
        client_config.experiment.experiment_name = (
            f"{base_config.experiment.experiment_name}_client_{client_id}"
        )
        client_config.logging.log_dir = (
            f"{base_config.logging.log_dir}/client_{client_id}"
        )

        return client_config

    def create_server_config(
        self, base_config: Optional[ZKFLConfig] = None
    ) -> ZKFLConfig:
        """
        Create server-specific configuration.

        Args:
            base_config: Base configuration to copy from

        Returns:
            Server-specific configuration
        """
        if base_config is None:
            base_config = self.get_config()

        # Create a copy for the server
        import copy

        server_config = copy.deepcopy(base_config)

        # Apply server-specific modifications
        server_config.experiment.experiment_name = (
            f"{base_config.experiment.experiment_name}_server"
        )
        server_config.logging.log_dir = f"{base_config.logging.log_dir}/server"

        return server_config

    def validate_environment(self) -> List[str]:
        """
        Validate the environment setup.

        Returns:
            List of validation warnings/errors
        """
        warnings = []

        config = self.get_config()

        # Check directory permissions
        directories_to_check = [
            config.data.data_path,
            config.logging.log_dir,
            config.experiment.output_dir,
        ]

        for directory in directories_to_check:
            dir_path = Path(directory)
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                # Test write access
                test_file = dir_path / ".test_write"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                warnings.append(f"Cannot write to directory {directory}: {e}")

        # Check SSL certificates if SSL is enabled
        if config.network.use_ssl:
            if (
                not config.network.ssl_cert_path
                or not Path(config.network.ssl_cert_path).exists()
            ):
                warnings.append("SSL enabled but certificate file not found")
            if (
                not config.network.ssl_key_path
                or not Path(config.network.ssl_key_path).exists()
            ):
                warnings.append("SSL enabled but key file not found")

        # Check blockchain configuration
        if config.blockchain.enable_blockchain:
            if not config.blockchain.network_url:
                warnings.append("Blockchain enabled but no network URL specified")
            if not config.blockchain.private_key:
                warnings.append("Blockchain enabled but no private key specified")

        return warnings

    def create_config_template(self, output_path: str):
        """
        Create a configuration template file.

        Args:
            output_path: Path for the template file
        """
        template_config = ZKFLConfig()
        self.save_config(template_config, output_path, "yaml")

        # Add comments to the template
        with open(output_path, "r") as f:
            content = f.read()

        # Add header comment
        header = """# ZKFL Configuration Template
# This file contains all available configuration options with their default values.
# Uncomment and modify values as needed for your specific setup.

"""

        with open(output_path, "w") as f:
            f.write(header + content)

    def get_config_diff(
        self, config1: ZKFLConfig, config2: ZKFLConfig
    ) -> Dict[str, Any]:
        """
        Get differences between two configurations.

        Args:
            config1: First configuration
            config2: Second configuration

        Returns:
            Dictionary of differences
        """
        diff = {}

        for field in fields(config1):
            section_name = field.name
            section1 = getattr(config1, section_name)
            section2 = getattr(config2, section_name)

            section_diff = {}
            for section_field in fields(section1):
                key = section_field.name
                value1 = getattr(section1, key)
                value2 = getattr(section2, key)

                if value1 != value2:
                    section_diff[key] = {"old": value1, "new": value2}

            if section_diff:
                diff[section_name] = section_diff

        return diff
