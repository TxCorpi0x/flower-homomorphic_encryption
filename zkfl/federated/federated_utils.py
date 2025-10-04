"""
Federated learning utilities for ZKFL.

This module provides utility functions and helpers for federated learning
operations, including client sampling, parameter aggregation, and metrics
computation.
"""

import logging
import random
import math
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class FederatedUtils:
    """Utility functions for federated learning operations."""

    @staticmethod
    def federated_averaging(
        parameter_updates: List[Tuple[Dict[str, Any], int]],
    ) -> Dict[str, Any]:
        """
        Perform federated averaging of parameter updates.

        Args:
            parameter_updates: List of (parameters, num_examples) tuples

        Returns:
            Aggregated parameters using federated averaging
        """
        if not parameter_updates:
            return {}

        # Calculate total number of examples
        total_examples = sum(num_examples for _, num_examples in parameter_updates)

        if total_examples == 0:
            logger.warning("Total examples is zero, using uniform averaging")
            return FederatedUtils.uniform_averaging(
                [params for params, _ in parameter_updates]
            )

        # Initialize aggregated parameters
        first_params, _ = parameter_updates[0]
        aggregated_params = {}

        for param_name in first_params.keys():
            aggregated_params[param_name] = 0

        # Compute weighted average
        for parameters, num_examples in parameter_updates:
            weight = num_examples / total_examples

            for param_name, param_value in parameters.items():
                if param_name in aggregated_params:
                    if isinstance(param_value, (int, float)):
                        aggregated_params[param_name] += weight * param_value
                    elif hasattr(param_value, "__mul__") and hasattr(
                        param_value, "__add__"
                    ):
                        # Handle arrays/tensors
                        if aggregated_params[param_name] == 0:
                            aggregated_params[param_name] = weight * param_value
                        else:
                            aggregated_params[param_name] = (
                                aggregated_params[param_name] + weight * param_value
                            )

        return aggregated_params

    @staticmethod
    def uniform_averaging(parameter_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform uniform averaging of parameters.

        Args:
            parameter_list: List of parameter dictionaries

        Returns:
            Uniformly averaged parameters
        """
        if not parameter_list:
            return {}

        # Initialize aggregated parameters
        aggregated_params = {}
        for param_name in parameter_list[0].keys():
            aggregated_params[param_name] = 0

        # Sum all parameters
        for parameters in parameter_list:
            for param_name, param_value in parameters.items():
                if param_name in aggregated_params:
                    if isinstance(param_value, (int, float)):
                        aggregated_params[param_name] += param_value
                    elif hasattr(param_value, "__add__"):
                        if aggregated_params[param_name] == 0:
                            aggregated_params[param_name] = param_value
                        else:
                            aggregated_params[param_name] = (
                                aggregated_params[param_name] + param_value
                            )

        # Divide by number of clients
        num_clients = len(parameter_list)
        for param_name in aggregated_params:
            if isinstance(aggregated_params[param_name], (int, float)):
                aggregated_params[param_name] /= num_clients
            elif hasattr(aggregated_params[param_name], "__truediv__"):
                aggregated_params[param_name] = (
                    aggregated_params[param_name] / num_clients
                )

        return aggregated_params

    @staticmethod
    def calculate_parameter_similarity(
        params1: Dict[str, Any], params2: Dict[str, Any]
    ) -> float:
        """
        Calculate cosine similarity between two parameter sets.

        Args:
            params1: First parameter set
            params2: Second parameter set

        Returns:
            Cosine similarity score (0-1)
        """
        if not params1 or not params2:
            return 0.0

        # Find common parameters
        common_params = set(params1.keys()) & set(params2.keys())

        if not common_params:
            return 0.0

        # Flatten parameters for similarity computation
        vec1, vec2 = [], []

        for param_name in common_params:
            val1, val2 = params1[param_name], params2[param_name]

            # Convert to flat list
            if isinstance(val1, (int, float)):
                vec1.append(val1)
                vec2.append(val2)
            elif hasattr(val1, "flatten"):
                # Array-like objects
                flat1 = val1.flatten() if hasattr(val1, "flatten") else [val1]
                flat2 = val2.flatten() if hasattr(val2, "flatten") else [val2]
                vec1.extend(flat1[:10])  # Limit to first 10 elements for efficiency
                vec2.extend(flat2[:10])

        if not vec1 or not vec2:
            return 0.0

        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def detect_byzantine_clients(
        parameter_updates: List[Tuple[str, Dict[str, Any]]], threshold: float = 0.5
    ) -> List[str]:
        """
        Detect potentially Byzantine (malicious) clients based on parameter similarity.

        Args:
            parameter_updates: List of (client_id, parameters) tuples
            threshold: Similarity threshold for Byzantine detection

        Returns:
            List of client IDs flagged as potentially Byzantine
        """
        if len(parameter_updates) < 3:
            return []  # Need at least 3 clients for meaningful detection

        byzantine_clients = []

        # Calculate pairwise similarities
        similarities = {}
        for i, (client_i, params_i) in enumerate(parameter_updates):
            similarities[client_i] = []

            for j, (client_j, params_j) in enumerate(parameter_updates):
                if i != j:
                    sim = FederatedUtils.calculate_parameter_similarity(
                        params_i, params_j
                    )
                    similarities[client_i].append(sim)

        # Flag clients with low average similarity
        for client_id, sim_scores in similarities.items():
            avg_similarity = sum(sim_scores) / len(sim_scores)
            if avg_similarity < threshold:
                byzantine_clients.append(client_id)
                logger.warning(
                    f"Client {client_id} flagged as potentially Byzantine (avg sim: {avg_similarity:.3f})"
                )

        return byzantine_clients

    @staticmethod
    def krum_aggregation(
        parameter_updates: List[Tuple[Dict[str, Any], int]], f: int = 1
    ) -> Dict[str, Any]:
        """
        Perform Krum aggregation for Byzantine-robust federated learning.

        Args:
            parameter_updates: List of (parameters, num_examples) tuples
            f: Number of Byzantine clients to tolerate

        Returns:
            Krum-aggregated parameters
        """
        if len(parameter_updates) < 2 * f + 3:
            logger.warning(
                "Insufficient clients for Krum aggregation, falling back to federated averaging"
            )
            return FederatedUtils.federated_averaging(parameter_updates)

        # Calculate Krum scores
        krum_scores = []
        for i, (params_i, _) in enumerate(parameter_updates):
            distances = []

            # Calculate distances to all other clients
            for j, (params_j, _) in enumerate(parameter_updates):
                if i != j:
                    # Simplified distance calculation
                    distance = 0.0
                    common_params = set(params_i.keys()) & set(params_j.keys())

                    for param_name in common_params:
                        val_i, val_j = params_i[param_name], params_j[param_name]
                        if isinstance(val_i, (int, float)) and isinstance(
                            val_j, (int, float)
                        ):
                            distance += (val_i - val_j) ** 2

                    distances.append(distance)

            # Krum score is sum of closest n-f-2 distances
            n = len(parameter_updates)
            closest_distances = sorted(distances)[: n - f - 2]
            krum_scores.append((i, sum(closest_distances)))

        # Select client with minimum Krum score
        best_client_idx = min(krum_scores, key=lambda x: x[1])[0]
        return parameter_updates[best_client_idx][0]

    @staticmethod
    def compute_global_metrics(
        client_metrics: List[Dict[str, Any]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute global metrics from client metrics.

        Args:
            client_metrics: List of metric dictionaries from clients
            client_weights: Optional weights for each client

        Returns:
            Aggregated global metrics
        """
        if not client_metrics:
            return {}

        if client_weights is None:
            client_weights = [1.0] * len(client_metrics)

        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Aggregate metrics
        global_metrics = {}

        # Find common metric keys
        common_keys = set(client_metrics[0].keys())
        for metrics in client_metrics[1:]:
            common_keys.intersection_update(metrics.keys())

        # Compute weighted averages
        for key in common_keys:
            values = []
            weights = []

            for i, metrics in enumerate(client_metrics):
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    values.append(value)
                    weights.append(normalized_weights[i])

            if values:
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                global_metrics[key] = weighted_sum

        # Add summary statistics
        global_metrics["num_clients"] = len(client_metrics)
        global_metrics["total_weight"] = total_weight

        return global_metrics

    @staticmethod
    def simulate_client_sampling(
        total_clients: int,
        participation_rate: float = 0.1,
        min_clients: int = 2,
        seed: Optional[int] = None,
    ) -> List[int]:
        """
        Simulate client sampling for federated learning rounds.

        Args:
            total_clients: Total number of available clients
            participation_rate: Fraction of clients to sample
            min_clients: Minimum number of clients to sample
            seed: Random seed for reproducibility

        Returns:
            List of selected client indices
        """
        if seed is not None:
            random.seed(seed)

        num_selected = max(min_clients, int(total_clients * participation_rate))
        num_selected = min(num_selected, total_clients)

        return random.sample(range(total_clients), num_selected)

    @staticmethod
    def calculate_communication_cost(
        parameters: Dict[str, Any], compression_ratio: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate communication cost for parameter transmission.

        Args:
            parameters: Parameter dictionary
            compression_ratio: Compression ratio (1.0 = no compression)

        Returns:
            Communication cost statistics
        """
        total_elements = 0
        total_size_bytes = 0

        for param_name, param_value in parameters.items():
            if isinstance(param_value, (int, float)):
                total_elements += 1
                total_size_bytes += 8  # Assume 8 bytes per float64
            elif hasattr(param_value, "size") and hasattr(param_value, "itemsize"):
                # NumPy-like arrays
                total_elements += param_value.size
                total_size_bytes += param_value.size * param_value.itemsize
            elif hasattr(param_value, "numel") and hasattr(param_value, "element_size"):
                # PyTorch-like tensors
                total_elements += param_value.numel()
                total_size_bytes += param_value.numel() * param_value.element_size()

        # Apply compression
        compressed_size = total_size_bytes * compression_ratio

        return {
            "total_elements": total_elements,
            "uncompressed_size_bytes": total_size_bytes,
            "compressed_size_bytes": compressed_size,
            "uncompressed_size_mb": total_size_bytes / (1024 * 1024),
            "compressed_size_mb": compressed_size / (1024 * 1024),
            "compression_ratio": compression_ratio,
            "bandwidth_saved_percent": (1 - compression_ratio) * 100,
        }


class ClientManager:
    """Manager for handling multiple federated learning clients."""

    def __init__(self):
        """Initialize client manager."""
        self.clients = {}
        self.client_statistics = defaultdict(dict)
        self.round_participation = defaultdict(list)

    def register_client(self, client_id: str, client: Any) -> None:
        """
        Register a client with the manager.

        Args:
            client_id: Unique client identifier
            client: Client instance
        """
        self.clients[client_id] = client
        self.client_statistics[client_id] = {
            "registration_time": "current_time",
            "total_rounds": 0,
            "last_active": None,
        }
        logger.info(f"Client {client_id} registered")

    def remove_client(self, client_id: str) -> bool:
        """
        Remove a client from the manager.

        Args:
            client_id: Client identifier to remove

        Returns:
            True if client was removed, False if not found
        """
        if client_id in self.clients:
            del self.clients[client_id]
            if client_id in self.client_statistics:
                del self.client_statistics[client_id]
            logger.info(f"Client {client_id} removed")
            return True
        return False

    def get_client(self, client_id: str) -> Optional[Any]:
        """Get a client by ID."""
        return self.clients.get(client_id)

    def all(self) -> List[Any]:
        """Get all registered clients."""
        return list(self.clients.values())

    def sample(self, num_clients: int, min_num_clients: int = 1) -> List[Any]:
        """
        Sample clients for a federated learning round.

        Args:
            num_clients: Number of clients to sample
            min_num_clients: Minimum number of clients required

        Returns:
            List of sampled clients
        """
        available_clients = list(self.clients.values())

        if len(available_clients) < min_num_clients:
            logger.warning(
                f"Insufficient clients: {len(available_clients)} < {min_num_clients}"
            )
            return []

        sample_size = min(num_clients, len(available_clients))
        sampled_clients = random.sample(available_clients, sample_size)

        # Update participation tracking
        client_ids = [
            cid for cid, client in self.clients.items() if client in sampled_clients
        ]
        current_round = max(self.round_participation.keys(), default=0) + 1
        self.round_participation[current_round] = client_ids

        return sampled_clients

    def update_client_statistics(
        self, client_id: str, round_metrics: Dict[str, Any]
    ) -> None:
        """Update statistics for a client after round completion."""
        if client_id in self.client_statistics:
            stats = self.client_statistics[client_id]
            stats["total_rounds"] += 1
            stats["last_active"] = "current_time"
            stats["last_metrics"] = round_metrics

    def get_participation_history(self) -> Dict[int, List[str]]:
        """Get client participation history by round."""
        return dict(self.round_participation)

    def get_client_statistics(self, client_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get client statistics.

        Args:
            client_id: Specific client ID, or None for all clients

        Returns:
            Client statistics
        """
        if client_id:
            return self.client_statistics.get(client_id, {})
        else:
            return dict(self.client_statistics)
