"""
Zero-knowledge proof system for federated learning.

This module implements ZK proofs to verify the integrity of model training
without revealing sensitive information about the training data or parameters.

Migrated and enhanced from going_modular/zk_security.py
"""

import json
import hashlib
import logging
import time
import pickle
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

from ..core.exceptions import ZKProofError, ConfigurationError

logger = logging.getLogger(__name__)


class ZKProofSystem:
    """
    Zero-knowledge proof system for federated learning verification.

    This class provides methods to generate and verify ZK proofs that demonstrate:
    1. Model was trained on declared data
    2. Training used specified hyperparameters
    3. Update follows legitimate learning algorithm
    4. No data leakage occurred
    5. Gradient bounds are respected
    """

    def __init__(
        self,
        circuit_path: Optional[str] = None,
        proving_key_path: Optional[str] = None,
        verification_key_path: Optional[str] = None,
    ):
        """
        Initialize ZK proof system.

        Args:
            circuit_path: Path to compiled circuit file
            proving_key_path: Path to proving key
            verification_key_path: Path to verification key
        """
        self.circuit_path = circuit_path or "./circuits/fl_circuit.json"
        self.proving_key_path = proving_key_path or "./keys/proving_key.json"
        self.verification_key_path = (
            verification_key_path or "./keys/verification_key.json"
        )

        # Initialize internal state
        self.proofs_generated = 0
        self.proofs_verified = 0
        self.proof_cache: Dict[str, Dict[str, Any]] = {}

        # Initialize mock ZK system (in real implementation, this would load actual ZK libraries)
        self._initialize_zk_system()

        logger.info("ZKProofSystem initialized")

    def _initialize_zk_system(self) -> None:
        """Initialize ZK proving system."""
        # In a real implementation, this would initialize libraries like:
        # - circomlib for circuit compilation
        # - snarkjs for proof generation
        # - arkworks for verification

        # For now, we'll use a mock system that demonstrates the concepts
        self.circuit_hash = hashlib.sha256(b"fl_training_circuit").hexdigest()
        logger.info(
            f"Initialized ZK system with circuit hash: {self.circuit_hash[:16]}..."
        )

    def generate_model_commitment(self, model_params: Dict[str, np.ndarray]) -> str:
        """
        Generate commitment hash for model parameters.

        Args:
            model_params: Dictionary of model parameters

        Returns:
            Commitment hash string
        """
        # Flatten and concatenate all parameters
        flattened_params = []
        param_names = sorted(model_params.keys())  # Ensure deterministic ordering

        for param_name in param_names:
            param_value = model_params[param_name]
            flattened_params.extend(param_value.flatten().tolist())

        # Create commitment hash
        param_str = json.dumps(flattened_params, sort_keys=True)
        commitment = hashlib.sha256(param_str.encode()).hexdigest()
        return commitment

    def generate_training_proof(
        self,
        old_model_params: Dict[str, np.ndarray],
        new_model_params: Dict[str, np.ndarray],
        training_data_hash: str,
        learning_rate: float,
        epochs: int,
        client_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate ZK proof that model update was computed correctly.

        Proves:
        1. Model was trained on declared data
        2. Training used specified hyperparameters
        3. Update follows legitimate learning algorithm
        4. No data leakage occurred
        5. Gradient bounds are respected

        Args:
            old_model_params: Previous model parameters
            new_model_params: Updated model parameters
            training_data_hash: Hash of training data
            learning_rate: Learning rate used
            epochs: Number of training epochs
            client_id: Client identifier

        Returns:
            Proof data dictionary
        """
        try:
            # Generate commitments
            old_commitment = self.generate_model_commitment(old_model_params)
            new_commitment = self.generate_model_commitment(new_model_params)

            # Compute gradient bounds for verification
            gradient_bounds = self._compute_gradient_bounds(
                old_model_params, new_model_params
            )

            # Prepare circuit inputs
            circuit_inputs = {
                "old_model_commitment": old_commitment,
                "new_model_commitment": new_commitment,
                "data_hash": training_data_hash,
                "learning_rate_scaled": int(
                    learning_rate * 1000000
                ),  # Scale for integer arithmetic
                "epochs": epochs,
                "gradient_bounds": gradient_bounds,
                "timestamp": int(time.time()),
                "client_id": client_id or "unknown",
            }

            # Generate proof (this would interface with actual ZK library)
            proof = self._generate_zk_proof(circuit_inputs)

            proof_data = {
                "proof": proof,
                "public_inputs": {
                    "old_commitment": old_commitment,
                    "new_commitment": new_commitment,
                    "data_hash": training_data_hash,
                    "timestamp": circuit_inputs["timestamp"],
                },
                "metadata": {
                    "client_id": client_id,
                    "round": None,  # To be set by FL system
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "gradient_norm": float(np.mean(gradient_bounds)),
                },
            }

            self.proofs_generated += 1
            logger.info(f"Generated ZK proof for client {client_id}")

            return proof_data

        except Exception as e:
            logger.error(f"Failed to generate ZK proof: {e}")
            raise ZKProofError(f"Proof generation failed: {e}")

    def verify_proof(self, proof_data: Dict[str, Any]) -> bool:
        """
        Verify a ZK proof.

        Args:
            proof_data: Proof data to verify

        Returns:
            True if proof is valid, False otherwise
        """
        try:
            # Interface with ZK verification system
            is_valid = self._verify_zk_proof(
                proof_data["proof"], proof_data["public_inputs"]
            )

            # Additional validity checks
            timestamp = proof_data["public_inputs"].get("timestamp", 0)
            current_time = int(time.time())

            # Proof shouldn't be too old (24 hours)
            if current_time - timestamp > 86400:
                logger.warning(f"Proof is too old ({current_time - timestamp} seconds)")
                return False

            self.proofs_verified += 1
            if is_valid:
                logger.info("✓ ZK proof verification passed")
            else:
                logger.warning("✗ ZK proof verification failed")

            return is_valid

        except Exception as e:
            logger.error(f"Error verifying ZK proof: {e}")
            return False

    def _compute_gradient_bounds(
        self, old_params: Dict, new_params: Dict
    ) -> List[float]:
        """Compute bounds on gradient magnitudes for proof verification."""
        bounds = []
        for key in sorted(old_params.keys()):
            if key in new_params:
                diff = new_params[key] - old_params[key]
                max_change = float(np.max(np.abs(diff)))
                bounds.append(max_change)
        return bounds

    def _generate_zk_proof(self, inputs: Dict) -> str:
        """Interface with ZK proving system."""
        # In a real implementation, this would:
        # 1. Compile the circuit with given inputs
        # 2. Generate witness
        # 3. Create proof using proving key
        # 4. Return proof as string/bytes

        # Mock proof generation
        inputs_str = json.dumps(inputs, sort_keys=True)
        proof_hash = hashlib.sha256(inputs_str.encode()).hexdigest()
        return f"zk_proof_{proof_hash[:32]}"

    def _verify_zk_proof(self, proof: str, public_inputs: Dict) -> bool:
        """Interface with ZK verification system."""
        # In a real implementation, this would:
        # 1. Parse the proof
        # 2. Verify using verification key
        # 3. Check public inputs

        # Mock verification - check proof format and consistency
        if not proof.startswith("zk_proof_"):
            return False

        # Verify proof matches public inputs
        inputs_str = json.dumps(public_inputs, sort_keys=True)
        expected_hash = hashlib.sha256(inputs_str.encode()).hexdigest()[:32]
        actual_hash = proof.replace("zk_proof_", "")

        return expected_hash == actual_hash

    def save_proof_data(self, file_path: str, proof_data: Dict[str, Any]) -> None:
        """
        Save ZK proof data to file.

        Args:
            file_path: Path to save proof data
            proof_data: Proof data to save
        """
        try:
            os.makedirs(
                os.path.dirname(file_path) if os.path.dirname(file_path) else ".",
                exist_ok=True,
            )
            with open(file_path, "wb") as f:
                pickle.dump(proof_data, f)
            logger.info(f"Saved proof data to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save proof data: {e}")
            raise ZKProofError(f"Cannot save proof: {e}")

    def load_proof_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load ZK proof data from file.

        Args:
            file_path: Path to load proof data from

        Returns:
            Proof data or None if not found
        """
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "rb") as f:
                proof_data = pickle.load(f)
            logger.info(f"Loaded proof data from {file_path}")
            return proof_data
        except Exception as e:
            logger.error(f"Error loading proof data: {e}")
            return None

    def create_training_data_hash(self, dataloader) -> str:
        """
        Create a hash of training data for ZK proof verification.

        Args:
            dataloader: Training data loader

        Returns:
            Hash string of training data
        """
        data_bytes = b""
        batch_count = 0

        # Sample first few batches for efficiency
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= 5:  # Only use first 5 batches
                break

            data_bytes += data.numpy().tobytes()
            data_bytes += target.numpy().tobytes()
            batch_count += 1

        # Include batch count and dataset size info
        metadata = f"batches_{batch_count}_size_{len(dataloader)}".encode()
        data_bytes += metadata

        return hashlib.sha256(data_bytes).hexdigest()

    def get_stats(self) -> Dict[str, int]:
        """Get proof generation and verification statistics."""
        return {
            "proofs_generated": self.proofs_generated,
            "proofs_verified": self.proofs_verified,
            "cached_proofs": len(self.proof_cache),
        }

    def generate_aggregation_proof(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        aggregated_params: Dict[str, np.ndarray],
        aggregation_weights: List[float],
        round_number: int,
    ) -> Dict[str, Any]:
        """
        Generate proof for federated aggregation correctness.

        Args:
            client_updates: List of client parameter updates
            aggregated_params: Final aggregated parameters
            aggregation_weights: Weights used for aggregation
            round_number: Current federated round

        Returns:
            Aggregation proof data
        """
        try:
            # Create commitments for all client updates
            client_commitments = [
                self.generate_model_commitment(update) for update in client_updates
            ]

            # Create commitment for aggregated result
            aggregated_commitment = self.generate_model_commitment(aggregated_params)

            # Prepare proof inputs
            aggregation_inputs = {
                "client_commitments": client_commitments,
                "aggregated_commitment": aggregated_commitment,
                "weights": aggregation_weights,
                "round": round_number,
                "timestamp": int(time.time()),
                "num_clients": len(client_updates),
            }

            # Generate aggregation proof
            proof = self._generate_zk_proof(aggregation_inputs)

            return {
                "proof": proof,
                "public_inputs": {
                    "aggregated_commitment": aggregated_commitment,
                    "num_clients": len(client_updates),
                    "round": round_number,
                    "timestamp": aggregation_inputs["timestamp"],
                },
                "metadata": {
                    "aggregation_method": "federated_averaging",
                    "weights_used": aggregation_weights,
                },
            }

        except Exception as e:
            logger.error(f"Failed to generate aggregation proof: {e}")
            raise ZKProofError(f"Aggregation proof generation failed: {e}")


# Utility functions for backward compatibility with going_modular/zk_security.py
def create_parameter_commitment(parameters: Dict[str, np.ndarray]) -> str:
    """Create commitment to model parameters (replaces FHE encryption)."""
    zk_prover = ZKProofSystem()
    return zk_prover.generate_model_commitment(parameters)


def save_proof_data(file_path: str, proof_data: Dict[str, Any]) -> None:
    """Save ZK proof data to file."""
    zk_system = ZKProofSystem()
    zk_system.save_proof_data(file_path, proof_data)


def load_proof_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Load ZK proof data from file."""
    zk_system = ZKProofSystem()
    return zk_system.load_proof_data(file_path)
