"""
ZKFL Production Package

This package contains production-ready federated learning scripts for deploying
the ZKFL framework in distributed environments using the Flower FL framework.

Components:
- server.py: Production FL server with ZK proofs, DP, and blockchain
- client.py: Production FL client with security features
- run_server.py: Server entry point script
- run_client.py: Client entry point script
"""

__version__ = "2.0.0"
__author__ = "ZKFL Team"

from .server import ZKFLProductionServer
from .client import ZKFLProductionClient

__all__ = [
    "ZKFLProductionServer",
    "ZKFLProductionClient",
]
