"""
Model management for the ZKFL framework.

This module provides utilities for managing neural network models,
including parameter extraction, serialization, and integrity verification.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import hashlib
import json

from .exceptions import ModelError, ValidationError


class ModelManager:
    """Manages neural network models for federated learning."""

    def __init__(self, model: Optional[nn.Module] = None):
        """
        Initialize model manager.

        Args:
            model: PyTorch model to manage
        """
        self._model = model
        self._parameter_history: List[Dict[str, np.ndarray]] = []
        self._integrity_hashes: List[str] = []

    @property
    def model(self) -> Optional[nn.Module]:
        """Get the managed model."""
        return self._model

    @model.setter
    def model(self, model: nn.Module) -> None:
        """Set the managed model."""
        if not isinstance(model, nn.Module):
            raise ModelError("Model must be a PyTorch nn.Module")
        self._model = model

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Extract model parameters as numpy arrays.

        Returns:
            Dictionary mapping parameter names to numpy arrays

        Raises:
            ModelError: If no model is set
        """
        if self._model is None:
            raise ModelError("No model set")

        parameters = {}
        for name, param in self._model.named_parameters():
            parameters[name] = param.cpu().detach().numpy()

        return parameters

    def set_parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Set model parameters from numpy arrays.

        Args:
            parameters: Dictionary mapping parameter names to numpy arrays

        Raises:
            ModelError: If no model is set or parameters are invalid
        """
        if self._model is None:
            raise ModelError("No model set")

        model_params = dict(self._model.named_parameters())

        for name, param_array in parameters.items():
            if name not in model_params:
                raise ModelError(f"Parameter '{name}' not found in model")

            param_tensor = torch.from_numpy(param_array)
            if param_tensor.shape != model_params[name].shape:
                raise ModelError(
                    f"Shape mismatch for parameter '{name}': "
                    f"expected {model_params[name].shape}, got {param_tensor.shape}"
                )

            model_params[name].data.copy_(param_tensor)

    def get_parameter_list(self) -> List[np.ndarray]:
        """
        Get model parameters as a list of numpy arrays.

        Returns:
            List of parameter arrays in consistent order
        """
        parameters = self.get_parameters()
        # Sort by parameter name for consistent ordering
        sorted_names = sorted(parameters.keys())
        return [parameters[name] for name in sorted_names]

    def set_parameter_list(self, parameter_list: List[np.ndarray]) -> None:
        """
        Set model parameters from a list of numpy arrays.

        Args:
            parameter_list: List of parameter arrays in consistent order
        """
        if self._model is None:
            raise ModelError("No model set")

        param_names = sorted(dict(self._model.named_parameters()).keys())

        if len(parameter_list) != len(param_names):
            raise ModelError(
                f"Parameter count mismatch: expected {len(param_names)}, "
                f"got {len(parameter_list)}"
            )

        parameters = dict(zip(param_names, parameter_list))
        self.set_parameters(parameters)

    def compute_parameter_hash(
        self, parameters: Optional[Dict[str, np.ndarray]] = None
    ) -> str:
        """
        Compute cryptographic hash of model parameters.

        Args:
            parameters: Parameters to hash. If None, uses current model parameters.

        Returns:
            Hexadecimal hash string
        """
        if parameters is None:
            parameters = self.get_parameters()

        # Create deterministic hash by sorting parameters and concatenating
        sorted_params = []
        for name in sorted(parameters.keys()):
            sorted_params.append(parameters[name].tobytes())

        combined_bytes = b"".join(sorted_params)
        return hashlib.sha256(combined_bytes).hexdigest()

    def save_model(self, path: Union[str, Path], include_metadata: bool = True) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
            include_metadata: Whether to include parameter metadata
        """
        if self._model is None:
            raise ModelError("No model set")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model_state_dict": self._model.state_dict(),
            "model_class": self._model.__class__.__name__,
        }

        if include_metadata:
            save_dict.update(
                {
                    "parameter_hash": self.compute_parameter_hash(),
                    "parameter_count": sum(p.numel() for p in self._model.parameters()),
                    "model_architecture": str(self._model),
                }
            )

        torch.save(save_dict, path)

    def load_model(
        self, path: Union[str, Path], model_class: Optional[type] = None
    ) -> None:
        """
        Load model from disk.

        Args:
            path: Path to load model from
            model_class: Model class for reconstruction (if not using existing model)
        """
        path = Path(path)
        if not path.exists():
            raise ModelError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")

        if self._model is None:
            if model_class is None:
                raise ModelError("No model set and no model class provided")
            self._model = model_class()

        self._model.load_state_dict(checkpoint["model_state_dict"])

    def validate_parameters(self, parameters: Dict[str, np.ndarray]) -> List[str]:
        """
        Validate parameter dictionary.

        Args:
            parameters: Parameters to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self._model is None:
            errors.append("No model set for validation")
            return errors

        model_params = dict(self._model.named_parameters())

        # Check for missing parameters
        missing_params = set(model_params.keys()) - set(parameters.keys())
        if missing_params:
            errors.append(f"Missing parameters: {missing_params}")

        # Check for extra parameters
        extra_params = set(parameters.keys()) - set(model_params.keys())
        if extra_params:
            errors.append(f"Extra parameters: {extra_params}")

        # Check parameter shapes and types
        for name, param_array in parameters.items():
            if name in model_params:
                expected_shape = model_params[name].shape
                if param_array.shape != expected_shape:
                    errors.append(
                        f"Shape mismatch for '{name}': "
                        f"expected {expected_shape}, got {param_array.shape}"
                    )

                if not isinstance(param_array, np.ndarray):
                    errors.append(f"Parameter '{name}' must be numpy array")
                elif not np.isfinite(param_array).all():
                    errors.append(f"Parameter '{name}' contains non-finite values")

        return errors

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary.

        Returns:
            Dictionary with model information
        """
        if self._model is None:
            return {"error": "No model set"}

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )

        return {
            "model_class": self._model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_hash": self.compute_parameter_hash(),
            "architecture": str(self._model),
            "device": next(self._model.parameters()).device.type,
        }

    def track_parameters(self) -> None:
        """Add current parameters to history for tracking."""
        if self._model is not None:
            parameters = self.get_parameters()
            self._parameter_history.append(parameters)
            self._integrity_hashes.append(self.compute_parameter_hash(parameters))

    def get_parameter_history(self) -> List[Dict[str, np.ndarray]]:
        """Get history of parameter snapshots."""
        return self._parameter_history.copy()

    def get_integrity_hashes(self) -> List[str]:
        """Get history of parameter integrity hashes."""
        return self._integrity_hashes.copy()

    def clear_history(self) -> None:
        """Clear parameter history."""
        self._parameter_history.clear()
        self._integrity_hashes.clear()

    def __repr__(self) -> str:
        """String representation of model manager."""
        if self._model is None:
            return "ModelManager(no model set)"

        total_params = sum(p.numel() for p in self._model.parameters())
        return f"ModelManager({self._model.__class__.__name__}, {total_params:,} parameters)"
