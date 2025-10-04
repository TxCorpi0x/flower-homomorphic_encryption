"""Federated learning components for ZKFL."""

from .flower_client import FlowerClient, ZKFLClient
from .flower_server import FlowerServer, ZKFLServer
from .federated_utils import FederatedUtils, ClientManager

# Convenient alias
AggregationStrategy = FederatedUtils

__all__ = [
    "FlowerClient",
    "FlowerServer",
    "ZKFLClient",
    "ZKFLServer",
    "FederatedUtils",
    "ClientManager",
    "AggregationStrategy",
]
