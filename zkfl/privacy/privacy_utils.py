"""
Privacy utilities and helper functions for ZKFL.

This module provides various privacy-preserving utilities including
secure multiparty computation helpers, homomorphic encryption utilities,
and privacy analysis tools.
"""

import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
import random
import math

from ..core.exceptions import PrivacyError
from ..crypto.crypto_utils import CryptoUtils

logger = logging.getLogger(__name__)


class PrivacyUtils:
    """Utility functions for privacy-preserving operations."""

    @staticmethod
    def compute_k_anonymity(
        dataset: List[Dict[str, Any]], quasi_identifiers: List[str], k: int = 5
    ) -> Dict[str, Any]:
        """
        Compute k-anonymity for a dataset.

        Args:
            dataset: List of data records
            quasi_identifiers: List of quasi-identifier column names
            k: Minimum group size for k-anonymity

        Returns:
            K-anonymity analysis results
        """
        if not dataset:
            return {"k_anonymous": True, "min_group_size": 0, "groups": []}

        # Group records by quasi-identifier values
        groups = {}
        for record in dataset:
            # Create key from quasi-identifiers
            key_values = []
            for qi in quasi_identifiers:
                key_values.append(str(record.get(qi, "")))
            key = "|".join(key_values)

            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        # Check k-anonymity
        group_sizes = [len(group) for group in groups.values()]
        min_group_size = min(group_sizes) if group_sizes else 0
        k_anonymous = min_group_size >= k

        return {
            "k_anonymous": k_anonymous,
            "min_group_size": min_group_size,
            "num_groups": len(groups),
            "groups": list(groups.keys()),
            "group_sizes": group_sizes,
        }

    @staticmethod
    def apply_l_diversity(
        dataset: List[Dict[str, Any]],
        quasi_identifiers: List[str],
        sensitive_attribute: str,
        l: int = 2,
    ) -> Dict[str, Any]:
        """
        Analyze l-diversity for a dataset.

        Args:
            dataset: List of data records
            quasi_identifiers: List of quasi-identifier column names
            sensitive_attribute: Name of sensitive attribute column
            l: Minimum number of distinct sensitive values per group

        Returns:
            L-diversity analysis results
        """
        if not dataset:
            return {"l_diverse": True, "min_diversity": 0, "groups": {}}

        # Group by quasi-identifiers
        groups = {}
        for record in dataset:
            key_values = []
            for qi in quasi_identifiers:
                key_values.append(str(record.get(qi, "")))
            key = "|".join(key_values)

            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        # Check l-diversity
        group_diversity = {}
        for group_key, records in groups.items():
            sensitive_values = set()
            for record in records:
                sensitive_values.add(str(record.get(sensitive_attribute, "")))
            group_diversity[group_key] = len(sensitive_values)

        min_diversity = min(group_diversity.values()) if group_diversity else 0
        l_diverse = min_diversity >= l

        return {
            "l_diverse": l_diverse,
            "min_diversity": min_diversity,
            "required_diversity": l,
            "groups": group_diversity,
        }

    @staticmethod
    def calculate_mutual_information(x_data: List[Any], y_data: List[Any]) -> float:
        """
        Calculate mutual information between two variables.

        Args:
            x_data: First variable data
            y_data: Second variable data

        Returns:
            Mutual information value
        """
        if len(x_data) != len(y_data):
            raise PrivacyError("Data arrays must have same length")

        # Convert to strings for consistent handling
        x_values = [str(x) for x in x_data]
        y_values = [str(y) for y in y_data]

        # Count frequencies
        n = len(x_values)
        x_counts = {}
        y_counts = {}
        xy_counts = {}

        for i in range(n):
            x_val, y_val = x_values[i], y_values[i]

            x_counts[x_val] = x_counts.get(x_val, 0) + 1
            y_counts[y_val] = y_counts.get(y_val, 0) + 1

            xy_key = f"{x_val}|{y_val}"
            xy_counts[xy_key] = xy_counts.get(xy_key, 0) + 1

        # Calculate mutual information
        mi = 0.0
        for xy_key, xy_count in xy_counts.items():
            x_val, y_val = xy_key.split("|")

            p_xy = xy_count / n
            p_x = x_counts[x_val] / n
            p_y = y_counts[y_val] / n

            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))

        return mi

    @staticmethod
    def generate_synthetic_data(
        original_data: List[Dict[str, Any]],
        num_synthetic: int,
        privacy_level: str = "medium",
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic data preserving privacy.

        Args:
            original_data: Original dataset
            num_synthetic: Number of synthetic records to generate
            privacy_level: Privacy level ("low", "medium", "high")

        Returns:
            List of synthetic data records
        """
        if not original_data:
            return []

        # Analyze data distributions
        columns = list(original_data[0].keys())
        column_stats = {}

        for col in columns:
            values = [record.get(col) for record in original_data]

            # Basic statistics
            if all(isinstance(v, (int, float)) for v in values if v is not None):
                # Numerical column
                numeric_values = [float(v) for v in values if v is not None]
                column_stats[col] = {
                    "type": "numeric",
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": sum(numeric_values) / len(numeric_values),
                    "values": numeric_values,
                }
            else:
                # Categorical column
                value_counts = {}
                for val in values:
                    if val is not None:
                        value_counts[str(val)] = value_counts.get(str(val), 0) + 1

                column_stats[col] = {
                    "type": "categorical",
                    "values": list(value_counts.keys()),
                    "probabilities": [
                        count / len(values) for count in value_counts.values()
                    ],
                }

        # Generate synthetic records
        synthetic_data = []
        noise_factor = {"low": 0.1, "medium": 0.2, "high": 0.5}.get(privacy_level, 0.2)

        for _ in range(num_synthetic):
            synthetic_record = {}

            for col, stats in column_stats.items():
                if stats["type"] == "numeric":
                    # Add noise to numeric values
                    base_value = random.choice(stats["values"])
                    noise_range = (stats["max"] - stats["min"]) * noise_factor
                    noise = random.uniform(-noise_range, noise_range)
                    synthetic_value = max(
                        stats["min"], min(stats["max"], base_value + noise)
                    )
                    synthetic_record[col] = synthetic_value
                else:
                    # Sample from categorical distribution
                    synthetic_value = random.choices(
                        stats["values"], weights=stats["probabilities"]
                    )[0]
                    synthetic_record[col] = synthetic_value

            synthetic_data.append(synthetic_record)

        return synthetic_data

    @staticmethod
    def mask_sensitive_fields(
        data: Dict[str, Any], sensitive_fields: List[str], masking_method: str = "hash"
    ) -> Dict[str, Any]:
        """
        Mask sensitive fields in data.

        Args:
            data: Data dictionary to mask
            sensitive_fields: List of field names to mask
            masking_method: Masking method ("hash", "redact", "pseudonymize")

        Returns:
            Data with masked sensitive fields
        """
        masked_data = data.copy()

        for field in sensitive_fields:
            if field in masked_data:
                original_value = str(masked_data[field])

                if masking_method == "hash":
                    # Hash the value
                    masked_data[field] = CryptoUtils.generate_secure_hash(
                        original_value
                    )[:16]
                elif masking_method == "redact":
                    # Replace with asterisks
                    masked_data[field] = "*" * min(len(original_value), 8)
                elif masking_method == "pseudonymize":
                    # Generate consistent pseudonym
                    seed = hash(original_value) % 1000000
                    masked_data[field] = f"USER_{seed:06d}"
                else:
                    # Default: partial masking
                    if len(original_value) > 4:
                        masked_data[field] = (
                            original_value[:2]
                            + "*" * (len(original_value) - 4)
                            + original_value[-2:]
                        )
                    else:
                        masked_data[field] = "*" * len(original_value)

        return masked_data

    @staticmethod
    def assess_reidentification_risk(
        dataset: List[Dict[str, Any]], quasi_identifiers: List[str]
    ) -> Dict[str, Any]:
        """
        Assess re-identification risk for a dataset.

        Args:
            dataset: Dataset to assess
            quasi_identifiers: List of quasi-identifier fields

        Returns:
            Re-identification risk assessment
        """
        if not dataset:
            return {"risk_level": "none", "unique_records": 0, "total_records": 0}

        # Count unique quasi-identifier combinations
        qi_combinations = set()
        for record in dataset:
            qi_values = tuple(str(record.get(qi, "")) for qi in quasi_identifiers)
            qi_combinations.add(qi_values)

        unique_records = len(qi_combinations)
        total_records = len(dataset)
        uniqueness_ratio = unique_records / total_records if total_records > 0 else 0

        # Assess risk level
        if uniqueness_ratio > 0.8:
            risk_level = "high"
        elif uniqueness_ratio > 0.5:
            risk_level = "medium"
        elif uniqueness_ratio > 0.2:
            risk_level = "low"
        else:
            risk_level = "very_low"

        return {
            "risk_level": risk_level,
            "unique_records": unique_records,
            "total_records": total_records,
            "uniqueness_ratio": uniqueness_ratio,
            "quasi_identifiers": quasi_identifiers,
        }

    @staticmethod
    def compute_information_loss(
        original_data: List[Dict[str, Any]],
        anonymized_data: List[Dict[str, Any]],
        numeric_fields: List[str],
    ) -> Dict[str, float]:
        """
        Compute information loss from anonymization.

        Args:
            original_data: Original dataset
            anonymized_data: Anonymized dataset
            numeric_fields: List of numeric field names

        Returns:
            Information loss metrics
        """
        if len(original_data) != len(anonymized_data):
            raise PrivacyError("Datasets must have same size")

        information_loss = {}

        for field in numeric_fields:
            original_values = []
            anonymized_values = []

            for i in range(len(original_data)):
                orig_val = original_data[i].get(field)
                anon_val = anonymized_data[i].get(field)

                if orig_val is not None and anon_val is not None:
                    original_values.append(float(orig_val))
                    anonymized_values.append(float(anon_val))

            if original_values:
                # Calculate metrics
                mean_abs_error = sum(
                    abs(o - a) for o, a in zip(original_values, anonymized_values)
                ) / len(original_values)
                original_range = max(original_values) - min(original_values)
                relative_error = mean_abs_error / max(original_range, 1e-10)

                information_loss[field] = {
                    "mean_absolute_error": mean_abs_error,
                    "relative_error": relative_error,
                    "data_utility": max(0, 1 - relative_error),
                }

        # Overall utility
        if information_loss:
            avg_utility = sum(
                metrics["data_utility"] for metrics in information_loss.values()
            ) / len(information_loss)
            information_loss["overall_utility"] = avg_utility

        return information_loss


class SecureAggregationHelper:
    """Helper for secure aggregation protocols."""

    def __init__(self):
        """Initialize secure aggregation helper."""
        self.participant_keys = {}
        self.shared_secrets = {}

    def generate_pairwise_keys(
        self, participants: List[str]
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate pairwise keys for secure aggregation.

        Args:
            participants: List of participant IDs

        Returns:
            Dictionary of pairwise keys
        """
        pairwise_keys = {}

        for i, participant_a in enumerate(participants):
            pairwise_keys[participant_a] = {}

            for j, participant_b in enumerate(participants):
                if i != j:
                    # Generate shared key
                    key_material = (
                        f"{participant_a}:{participant_b}:{random.randint(0, 1000000)}"
                    )
                    shared_key = CryptoUtils.generate_secure_hash(key_material)[:32]
                    pairwise_keys[participant_a][participant_b] = shared_key

        return pairwise_keys

    def create_secret_shares(
        self, value: float, num_shares: int, threshold: int
    ) -> List[Tuple[int, float]]:
        """
        Create secret shares using Shamir's secret sharing.

        Args:
            value: Secret value to share
            num_shares: Total number of shares
            threshold: Minimum shares needed to reconstruct

        Returns:
            List of (share_id, share_value) tuples
        """
        # Simplified secret sharing (for demonstration)
        # In practice, use proper finite field arithmetic

        # Generate random coefficients for polynomial
        coefficients = [value]  # constant term is the secret
        for _ in range(threshold - 1):
            coefficients.append(random.uniform(-1000, 1000))

        # Generate shares
        shares = []
        for i in range(1, num_shares + 1):
            # Evaluate polynomial at point i
            share_value = sum(
                coeff * (i**power) for power, coeff in enumerate(coefficients)
            )
            shares.append((i, share_value))

        return shares

    def reconstruct_secret(self, shares: List[Tuple[int, float]]) -> float:
        """
        Reconstruct secret from shares using Lagrange interpolation.

        Args:
            shares: List of (share_id, share_value) tuples

        Returns:
            Reconstructed secret value
        """
        if not shares:
            return 0.0

        # Lagrange interpolation at x=0
        secret = 0.0

        for i, (x_i, y_i) in enumerate(shares):
            # Compute Lagrange basis polynomial
            basis = 1.0
            for j, (x_j, _) in enumerate(shares):
                if i != j:
                    basis *= (0 - x_j) / (x_i - x_j)

            secret += y_i * basis

        return secret
