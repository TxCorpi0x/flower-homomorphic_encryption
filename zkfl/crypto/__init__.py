"""Zero-knowledge proof cryptographic module for ZKFL."""

from .zk_proof import ZKProofSystem
from .crypto_utils import CryptoUtils, SecureAggregator, DigitalSignature

# Convenient aliases
CryptographicUtils = CryptoUtils
CommitmentScheme = CryptoUtils

__all__ = [
    "ZKProofSystem",
    "CryptoUtils",
    "CryptographicUtils",
    "CommitmentScheme",
    "SecureAggregator",
    "DigitalSignature",
]
