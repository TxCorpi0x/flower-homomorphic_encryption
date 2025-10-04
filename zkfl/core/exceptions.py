"""
Custom exceptions for the ZKFL framework.

This module defines all custom exceptions used throughout the framework
to provide clear error handling and debugging information.
"""

from typing import Optional, Any


class ZKFLError(Exception):
    """Base exception class for all ZKFL-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details
        self.message = message

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ZKProofError(ZKFLError):
    """Exception raised for zero-knowledge proof related errors."""

    def __init__(self, message: str, proof_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="ZK_ERROR", **kwargs)
        self.proof_type = proof_type


class BlockchainError(ZKFLError):
    """Exception raised for blockchain-related errors."""

    def __init__(self, message: str, transaction_hash: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="BLOCKCHAIN_ERROR", **kwargs)
        self.transaction_hash = transaction_hash


class SmartContractError(ZKFLError):
    """Exception raised for smart contract related errors."""

    def __init__(self, message: str, contract_address: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="SMART_CONTRACT_ERROR", **kwargs)
        self.contract_address = contract_address


class PrivacyError(ZKFLError):
    """Exception raised for differential privacy related errors."""

    def __init__(self, message: str, privacy_budget: Optional[float] = None, **kwargs):
        super().__init__(message, error_code="PRIVACY_ERROR", **kwargs)
        self.privacy_budget = privacy_budget


class ModelError(ZKFLError):
    """Exception raised for model-related errors."""

    def __init__(self, message: str, model_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)
        self.model_type = model_type


class ModelBuildingError(ModelError):
    """Exception raised for model building specific errors."""

    def __init__(self, message: str, architecture: Optional[str] = None, **kwargs):
        # Don't pass error_code since parent already sets it
        super().__init__(message, model_type=architecture, **kwargs)
        self.architecture = architecture


class ModelOperationError(ModelError):
    """Exception raised for model operation specific errors."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        # Don't pass error_code since parent already sets it
        super().__init__(message, model_type=operation, **kwargs)
        self.operation = operation


class FederatedLearningError(ZKFLError):
    """Exception raised for federated learning specific errors."""

    def __init__(
        self,
        message: str,
        round_number: Optional[int] = None,
        client_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="FL_ERROR", **kwargs)
        self.round_number = round_number
        self.client_id = client_id


class ConfigurationError(ZKFLError):
    """Exception raised for configuration-related errors."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key


class ValidationError(ZKFLError):
    """Exception raised for data validation errors."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field


class DataProcessingError(ZKFLError):
    """Exception raised for data processing errors."""

    def __init__(self, message: str, dataset: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DATA_ERROR", **kwargs)
        self.dataset = dataset


class CryptographicError(ZKFLError):
    """Exception raised for general cryptographic errors."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CRYPTO_ERROR", **kwargs)
        self.operation = operation


class MetricsError(ZKFLError):
    """Exception raised for metrics computation errors."""

    def __init__(self, message: str, metric_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="METRICS_ERROR", **kwargs)
        self.metric_name = metric_name
