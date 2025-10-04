"""
ZKFL: Zero-Knowledge Federated Learning Framework

A modern federated learning framework that combines zero-knowledge proofs,
differential privacy, and blockchain technology for secure, transparent,
and verifiable machine learning.

Features:
- Zero-knowledge proof verification of training integrity
- Differential privacy for parameter protection
- Blockchain integration for transparency and auditability
- Modular architecture with clean APIs
- Production-ready components

Example:
    Basic usage:

    ```python
    from zkfl.core import FederatedLearningEngine
    from zkfl.crypto import ZKProofSystem
    from zkfl.privacy import DifferentialPrivacyManager

    # Initialize components
    zk_system = ZKProofSystem()
    dp_manager = DifferentialPrivacyManager(epsilon=1.0)
    fl_engine = FederatedLearningEngine(zk_system, dp_manager)

    # Start federated learning
    fl_engine.start_training()
    ```
"""

__version__ = "2.0.0"
__author__ = "ZKFL Development Team"
__email__ = "contact@zkfl.org"
__license__ = "MIT"

# Core exports
from zkfl.core import (
    FederatedLearningEngine,
    ModelManager,
    TrainingEngine,
)

from zkfl.crypto import (
    ZKProofSystem,
    CommitmentScheme,
    CryptographicUtils,
)

from zkfl.privacy import (
    DifferentialPrivacyManager,
    NoiseGenerator,
    PrivacyAccountant,
)

from zkfl.blockchain import (
    BlockchainInterface,
    SmartContractManager,
    TransactionHandler,
)

from zkfl.federated import (
    FlowerClient,
    FlowerServer,
    AggregationStrategy,
)

# Convenience imports
from zkfl.models import get_model
from zkfl.data import get_dataset, DataLoader
from zkfl.utils import LoggingUtils, ConfigUtils

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Core components
    "FederatedLearningEngine",
    "ModelManager",
    "TrainingEngine",
    # Cryptographic components
    "ZKProofSystem",
    "CommitmentScheme",
    "CryptographicUtils",
    # Privacy components
    "DifferentialPrivacyManager",
    "NoiseGenerator",
    "PrivacyAccountant",
    # Blockchain components
    "BlockchainInterface",
    "SmartContractManager",
    "TransactionHandler",
    # Federated learning components
    "FlowerClient",
    "FlowerServer",
    "AggregationStrategy",
    # Utilities
    "get_model",
    "get_dataset",
    "DataLoader",
    "LoggingUtils",
    "ConfigUtils",
]

# Setup logging for the package
import logging

# Configure package-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info(f"ZKFL v{__version__} initialized")
