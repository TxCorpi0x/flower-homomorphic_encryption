"""Blockchain integration module for ZKFL."""

from .blockchain_interface import BlockchainInterface
from .smart_contract import SmartContractManager

# Convenient alias
TransactionHandler = BlockchainInterface

__all__ = ["BlockchainInterface", "SmartContractManager", "TransactionHandler"]
