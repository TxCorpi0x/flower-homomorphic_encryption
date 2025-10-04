"""
Differential privacy implementation for federated learning.

This module provides differential privacy mechanisms to protect individual
client data while maintaining model utility in federated learning.
"""

import math
import random
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from ..core.exceptions import PrivacyError, ConfigurationError

logger = logging.getLogger(__name__)


class DifferentialPrivacyManager:
    """Manager for differential privacy in federated learning."""

    def __init__(
        self,
        noise_scale: float = 1.0,
        clip_norm: float = 1.0,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
    ):
        """
        Initialize differential privacy manager.

        Args:
            noise_scale: Scale parameter for noise distribution
            clip_norm: L2 norm clipping threshold
            epsilon: Privacy parameter (smaller = more private)
            delta: Failure probability (should be small)
            sensitivity: Global sensitivity of the mechanism
        """
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

        # Privacy accounting
        self.privacy_budget_used = 0.0
        self.noise_added_count = 0
        self.clipping_applied_count = 0

        # Validate parameters
        self._validate_parameters()

        logger.info(
            f"DifferentialPrivacyManager initialized with ε={epsilon}, δ={delta}"
        )

    def _validate_parameters(self) -> None:
        """Validate differential privacy parameters."""
        if self.epsilon <= 0:
            raise ConfigurationError("Epsilon must be positive")

        if self.delta < 0 or self.delta >= 1:
            raise ConfigurationError("Delta must be in [0, 1)")

        if self.noise_scale <= 0:
            raise ConfigurationError("Noise scale must be positive")

        if self.clip_norm <= 0:
            raise ConfigurationError("Clip norm must be positive")

    def add_noise_to_parameters(
        self, parameters: Dict[str, Any], mechanism: str = "gaussian"
    ) -> Dict[str, Any]:
        """
        Add differential privacy noise to model parameters.

        Args:
            parameters: Model parameters to protect
            mechanism: Noise mechanism ("gaussian" or "laplace")

        Returns:
            Parameters with added noise

        Raises:
            PrivacyError: If noise addition fails
        """
        try:
            logger.info(f"Adding {mechanism} noise to model parameters")

            # Check privacy budget
            if not self._check_privacy_budget():
                raise PrivacyError("Privacy budget exhausted")

            # Clip parameters first
            clipped_params = self._clip_parameters(parameters)

            # Add noise
            if mechanism == "gaussian":
                noisy_params = self._add_gaussian_noise(clipped_params)
            elif mechanism == "laplace":
                noisy_params = self._add_laplace_noise(clipped_params)
            else:
                raise PrivacyError(f"Unknown noise mechanism: {mechanism}")

            # Update privacy accounting
            self._update_privacy_budget(mechanism)
            self.noise_added_count += 1

            logger.info("Differential privacy noise added successfully")
            return noisy_params

        except Exception as e:
            logger.error(f"Failed to add DP noise: {e}")
            raise PrivacyError(f"Noise addition failed: {e}")

    def _clip_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply gradient clipping to parameters."""
        clipped_params = {}

        for param_name, param_value in parameters.items():
            if isinstance(param_value, (list, tuple)):
                # Convert to numpy array for processing
                param_array = np.array(param_value)
            elif hasattr(param_value, "numpy"):
                # PyTorch tensor
                param_array = param_value.numpy()
            elif isinstance(param_value, np.ndarray):
                param_array = param_value
            else:
                # Try to convert to float array
                param_array = np.array([float(param_value)])

            # Compute L2 norm
            param_norm = np.linalg.norm(param_array)

            # Apply clipping if needed
            if param_norm > self.clip_norm:
                clipped_array = param_array * (self.clip_norm / param_norm)
                self.clipping_applied_count += 1
                logger.debug(
                    f"Clipped parameter {param_name}: {param_norm:.4f} -> {self.clip_norm}"
                )
            else:
                clipped_array = param_array

            # Convert back to original format
            if isinstance(param_value, (list, tuple)):
                clipped_params[param_name] = clipped_array.tolist()
            else:
                clipped_params[param_name] = clipped_array

        return clipped_params

    def _add_gaussian_noise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add Gaussian noise to parameters."""
        noisy_params = {}

        for param_name, param_value in parameters.items():
            if isinstance(param_value, (list, tuple)):
                param_array = np.array(param_value)
            elif hasattr(param_value, "numpy"):
                param_array = param_value.numpy()
            elif isinstance(param_value, np.ndarray):
                param_array = param_value
            else:
                param_array = np.array([float(param_value)])

            # Calculate noise standard deviation
            # σ = (sensitivity * noise_scale) / ε for (ε,δ)-DP
            noise_std = (self.sensitivity * self.noise_scale) / self.epsilon

            # Generate Gaussian noise
            noise = np.random.normal(0, noise_std, param_array.shape)
            noisy_array = param_array + noise

            # Convert back to original format
            if isinstance(param_value, (list, tuple)):
                noisy_params[param_name] = noisy_array.tolist()
            else:
                noisy_params[param_name] = noisy_array

        return noisy_params

    def _add_laplace_noise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add Laplace noise to parameters."""
        noisy_params = {}

        for param_name, param_value in parameters.items():
            if isinstance(param_value, (list, tuple)):
                param_array = np.array(param_value)
            elif hasattr(param_value, "numpy"):
                param_array = param_value.numpy()
            elif isinstance(param_value, np.ndarray):
                param_array = param_value
            else:
                param_array = np.array([float(param_value)])

            # Calculate noise scale parameter
            # b = sensitivity / ε for ε-DP
            noise_scale = self.sensitivity / self.epsilon

            # Generate Laplace noise
            noise = np.random.laplace(0, noise_scale, param_array.shape)
            noisy_array = param_array + noise

            # Convert back to original format
            if isinstance(param_value, (list, tuple)):
                noisy_params[param_name] = noisy_array.tolist()
            else:
                noisy_params[param_name] = noisy_array

        return noisy_params

    def _check_privacy_budget(self) -> bool:
        """Check if privacy budget allows for more operations."""
        # Simple budget check - in practice, this would be more sophisticated
        return self.privacy_budget_used < self.epsilon

    def _update_privacy_budget(self, mechanism: str) -> None:
        """Update privacy budget accounting."""
        # Simple budget update - real implementation would use composition theorems
        if mechanism == "gaussian":
            # Gaussian mechanism with (ε,δ)-DP
            budget_used = self.epsilon / 10  # Conservative estimate
        elif mechanism == "laplace":
            # Laplace mechanism with ε-DP
            budget_used = self.epsilon / 20  # Conservative estimate
        else:
            budget_used = self.epsilon / 5

        self.privacy_budget_used += budget_used
        logger.debug(
            f"Privacy budget used: {self.privacy_budget_used:.4f}/{self.epsilon}"
        )

    def compute_privacy_cost(
        self, num_rounds: int, clients_per_round: int, mechanism: str = "gaussian"
    ) -> Dict[str, float]:
        """
        Compute total privacy cost for federated learning.

        Args:
            num_rounds: Number of FL rounds
            clients_per_round: Average clients per round
            mechanism: Noise mechanism

        Returns:
            Privacy cost analysis
        """
        # Simplified privacy analysis
        operations_per_round = clients_per_round
        total_operations = num_rounds * operations_per_round

        if mechanism == "gaussian":
            # Using advanced composition for Gaussian mechanism
            cost_per_operation = self.epsilon / (
                2 * math.sqrt(2 * math.log(1.25 / self.delta))
            )
        else:
            # Basic composition for other mechanisms
            cost_per_operation = self.epsilon / total_operations

        total_cost = total_operations * cost_per_operation

        return {
            "total_epsilon_cost": total_cost,
            "cost_per_round": cost_per_operation * operations_per_round,
            "cost_per_client": cost_per_operation,
            "remaining_budget": max(0, self.epsilon - total_cost),
            "feasible": total_cost <= self.epsilon,
        }

    def add_noise_to_aggregation(
        self, aggregated_params: Dict[str, Any], num_clients: int
    ) -> Dict[str, Any]:
        """
        Add noise to aggregated parameters at server.

        Args:
            aggregated_params: Aggregated model parameters
            num_clients: Number of clients that contributed

        Returns:
            Noisy aggregated parameters
        """
        logger.info(f"Adding DP noise to aggregation from {num_clients} clients")

        # Adjust noise scale based on number of clients
        original_scale = self.noise_scale
        self.noise_scale = self.noise_scale / math.sqrt(num_clients)

        try:
            noisy_params = self.add_noise_to_parameters(
                aggregated_params, mechanism="gaussian"
            )
        finally:
            # Restore original noise scale
            self.noise_scale = original_scale

        return noisy_params

    def analyze_privacy_leakage(
        self, original_params: Dict[str, Any], noisy_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Analyze privacy leakage from noisy parameters.

        Args:
            original_params: Original parameters
            noisy_params: Parameters with added noise

        Returns:
            Privacy leakage analysis
        """
        leakage_metrics = {}

        for param_name in original_params.keys():
            if param_name not in noisy_params:
                continue

            orig_array = np.array(original_params[param_name])
            noisy_array = np.array(noisy_params[param_name])

            # Compute noise magnitude
            noise = noisy_array - orig_array
            noise_magnitude = np.linalg.norm(noise)
            signal_magnitude = np.linalg.norm(orig_array)

            # Signal-to-noise ratio
            snr = signal_magnitude / max(noise_magnitude, 1e-10)

            # Relative error
            relative_error = noise_magnitude / max(signal_magnitude, 1e-10)

            leakage_metrics[param_name] = {
                "noise_magnitude": float(noise_magnitude),
                "signal_magnitude": float(signal_magnitude),
                "snr": float(snr),
                "relative_error": float(relative_error),
            }

        # Overall metrics
        avg_snr = np.mean([m["snr"] for m in leakage_metrics.values()])
        avg_error = np.mean([m["relative_error"] for m in leakage_metrics.values()])

        return {
            "per_parameter": leakage_metrics,
            "average_snr": avg_snr,
            "average_relative_error": avg_error,
            "privacy_level": (
                "high" if avg_error > 0.1 else "medium" if avg_error > 0.01 else "low"
            ),
        }

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.epsilon - self.privacy_budget_used)

    def get_statistics(self) -> Dict[str, Any]:
        """Get differential privacy statistics."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_scale": self.noise_scale,
            "clip_norm": self.clip_norm,
            "privacy_budget_used": self.privacy_budget_used,
            "remaining_budget": self.get_remaining_budget(),
            "noise_operations": self.noise_added_count,
            "clipping_operations": self.clipping_applied_count,
            "budget_utilization": (self.privacy_budget_used / self.epsilon) * 100,
        }

    def reset_budget(self) -> None:
        """Reset privacy budget accounting."""
        self.privacy_budget_used = 0.0
        self.noise_added_count = 0
        self.clipping_applied_count = 0
        logger.info("Privacy budget reset")

    def update_parameters(
        self,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        noise_scale: Optional[float] = None,
        clip_norm: Optional[float] = None,
    ) -> None:
        """
        Update differential privacy parameters.

        Args:
            epsilon: New epsilon value
            delta: New delta value
            noise_scale: New noise scale
            clip_norm: New clipping norm
        """
        if epsilon is not None:
            self.epsilon = epsilon
        if delta is not None:
            self.delta = delta
        if noise_scale is not None:
            self.noise_scale = noise_scale
        if clip_norm is not None:
            self.clip_norm = clip_norm

        # Revalidate parameters
        self._validate_parameters()

        logger.info("Differential privacy parameters updated")


class LocalDifferentialPrivacy:
    """Local differential privacy for client-side protection."""

    def __init__(self, epsilon: float = 1.0):
        """
        Initialize local differential privacy.

        Args:
            epsilon: Privacy parameter for local DP
        """
        self.epsilon = epsilon
        self.randomization_count = 0

    def randomize_response(
        self, true_value: bool, probability: Optional[float] = None
    ) -> bool:
        """
        Apply randomized response mechanism.

        Args:
            true_value: True boolean value
            probability: Probability of truthful response (computed if None)

        Returns:
            Randomized boolean response
        """
        if probability is None:
            # Optimal probability for binary randomized response
            probability = math.exp(self.epsilon) / (1 + math.exp(self.epsilon))

        self.randomization_count += 1

        if random.random() < probability:
            return true_value
        else:
            return not true_value

    def add_local_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """
        Add local differential privacy noise.

        Args:
            value: Original value
            sensitivity: Sensitivity of the value

        Returns:
            Value with local DP noise
        """
        # Use Laplace mechanism for local DP
        noise_scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, noise_scale)

        self.randomization_count += 1
        return value + noise

    def get_statistics(self) -> Dict[str, Any]:
        """Get local DP statistics."""
        return {
            "epsilon": self.epsilon,
            "randomizations": self.randomization_count,
            "truthful_probability": math.exp(self.epsilon)
            / (1 + math.exp(self.epsilon)),
        }
