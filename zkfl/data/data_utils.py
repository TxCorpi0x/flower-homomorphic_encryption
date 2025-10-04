"""
Data utilities for ZKFL federated learning.

This module provides utility functions for data preprocessing,
augmentation, and analysis in federated learning contexts.
"""

import logging
import random
import math
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from collections import Counter

from ..core.exceptions import DataProcessingError

logger = logging.getLogger(__name__)


class DataUtils:
    """Utility functions for data processing in federated learning."""

    @staticmethod
    def normalize_data(
        data: List[List[float]],
        method: str = "minmax",
        feature_range: Tuple[float, float] = (0, 1),
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Normalize data using specified method.

        Args:
            data: Input data as list of samples
            method: Normalization method ("minmax", "standard", "robust")
            feature_range: Target range for minmax scaling

        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        if not data or not data[0]:
            return data, {}

        num_features = len(data[0])
        normalized_data = []
        normalization_params = {"method": method}

        if method == "minmax":
            # Calculate min and max for each feature
            feature_mins = [float("inf")] * num_features
            feature_maxs = [float("-inf")] * num_features

            for sample in data:
                for i, value in enumerate(sample):
                    feature_mins[i] = min(feature_mins[i], value)
                    feature_maxs[i] = max(feature_maxs[i], value)

            normalization_params["feature_mins"] = feature_mins
            normalization_params["feature_maxs"] = feature_maxs
            normalization_params["feature_range"] = feature_range

            # Normalize samples
            for sample in data:
                normalized_sample = []
                for i, value in enumerate(sample):
                    if feature_maxs[i] != feature_mins[i]:
                        # Scale to [0, 1] then to target range
                        scaled = (value - feature_mins[i]) / (
                            feature_maxs[i] - feature_mins[i]
                        )
                        normalized_value = feature_range[0] + scaled * (
                            feature_range[1] - feature_range[0]
                        )
                    else:
                        normalized_value = feature_range[0]
                    normalized_sample.append(normalized_value)
                normalized_data.append(normalized_sample)

        elif method == "standard":
            # Calculate mean and std for each feature
            feature_means = [0.0] * num_features
            feature_stds = [0.0] * num_features

            # Calculate means
            for sample in data:
                for i, value in enumerate(sample):
                    feature_means[i] += value

            for i in range(num_features):
                feature_means[i] /= len(data)

            # Calculate standard deviations
            for sample in data:
                for i, value in enumerate(sample):
                    feature_stds[i] += (value - feature_means[i]) ** 2

            for i in range(num_features):
                feature_stds[i] = math.sqrt(feature_stds[i] / len(data))

            normalization_params["feature_means"] = feature_means
            normalization_params["feature_stds"] = feature_stds

            # Normalize samples
            for sample in data:
                normalized_sample = []
                for i, value in enumerate(sample):
                    if feature_stds[i] != 0:
                        normalized_value = (value - feature_means[i]) / feature_stds[i]
                    else:
                        normalized_value = 0.0
                    normalized_sample.append(normalized_value)
                normalized_data.append(normalized_sample)

        else:
            raise DataProcessingError(f"Unknown normalization method: {method}")

        logger.info(f"Normalized {len(data)} samples using {method} method")
        return normalized_data, normalization_params

    @staticmethod
    def apply_normalization(
        data: List[List[float]], normalization_params: Dict[str, Any]
    ) -> List[List[float]]:
        """
        Apply previously computed normalization parameters to new data.

        Args:
            data: Data to normalize
            normalization_params: Parameters from previous normalization

        Returns:
            Normalized data
        """
        method = normalization_params.get("method")

        if method == "minmax":
            feature_mins = normalization_params["feature_mins"]
            feature_maxs = normalization_params["feature_maxs"]
            feature_range = normalization_params["feature_range"]

            normalized_data = []
            for sample in data:
                normalized_sample = []
                for i, value in enumerate(sample):
                    if i < len(feature_mins) and feature_maxs[i] != feature_mins[i]:
                        scaled = (value - feature_mins[i]) / (
                            feature_maxs[i] - feature_mins[i]
                        )
                        normalized_value = feature_range[0] + scaled * (
                            feature_range[1] - feature_range[0]
                        )
                    else:
                        normalized_value = (
                            feature_range[0] if len(feature_range) > 0 else 0.0
                        )
                    normalized_sample.append(normalized_value)
                normalized_data.append(normalized_sample)

            return normalized_data

        elif method == "standard":
            feature_means = normalization_params["feature_means"]
            feature_stds = normalization_params["feature_stds"]

            normalized_data = []
            for sample in data:
                normalized_sample = []
                for i, value in enumerate(sample):
                    if i < len(feature_means) and feature_stds[i] != 0:
                        normalized_value = (value - feature_means[i]) / feature_stds[i]
                    else:
                        normalized_value = 0.0
                    normalized_sample.append(normalized_value)
                normalized_data.append(normalized_sample)

            return normalized_data

        else:
            return data

    @staticmethod
    def augment_data(
        data: List[List[float]],
        labels: List[int],
        augmentation_factor: float = 1.5,
        noise_level: float = 0.1,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Augment data with noise and transformations.

        Args:
            data: Original data
            labels: Original labels
            augmentation_factor: Factor by which to increase dataset size
            noise_level: Standard deviation of Gaussian noise to add

        Returns:
            Tuple of (augmented_data, augmented_labels)
        """
        if augmentation_factor <= 1.0:
            return data, labels

        original_size = len(data)
        target_size = int(original_size * augmentation_factor)
        additional_samples = target_size - original_size

        augmented_data = data.copy()
        augmented_labels = labels.copy()

        for _ in range(additional_samples):
            # Select random original sample
            idx = random.randint(0, original_size - 1)
            original_sample = data[idx]
            original_label = labels[idx]

            # Create augmented sample
            augmented_sample = []
            for value in original_sample:
                # Add Gaussian noise
                noise = random.gauss(0, noise_level)
                augmented_value = value + noise
                augmented_sample.append(augmented_value)

            augmented_data.append(augmented_sample)
            augmented_labels.append(original_label)

        logger.info(
            f"Augmented dataset from {original_size} to {len(augmented_data)} samples"
        )
        return augmented_data, augmented_labels

    @staticmethod
    def balance_dataset(
        data: List[List[float]], labels: List[int], method: str = "oversample"
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Balance dataset across classes.

        Args:
            data: Input data
            labels: Input labels
            method: Balancing method ("oversample", "undersample")

        Returns:
            Tuple of (balanced_data, balanced_labels)
        """
        # Count samples per class
        class_counts = Counter(labels)

        if method == "oversample":
            # Oversample to match the majority class
            max_count = max(class_counts.values())

            balanced_data = []
            balanced_labels = []

            # Group data by class
            class_data = {}
            for i, label in enumerate(labels):
                if label not in class_data:
                    class_data[label] = []
                class_data[label].append((data[i], label))

            # Oversample each class
            for class_label, samples in class_data.items():
                class_samples = [sample[0] for sample in samples]
                class_labels = [sample[1] for sample in samples]

                # Add original samples
                balanced_data.extend(class_samples)
                balanced_labels.extend(class_labels)

                # Oversample to reach max_count
                samples_needed = max_count - len(class_samples)
                for _ in range(samples_needed):
                    idx = random.randint(0, len(class_samples) - 1)
                    balanced_data.append(class_samples[idx])
                    balanced_labels.append(class_labels[idx])

        elif method == "undersample":
            # Undersample to match the minority class
            min_count = min(class_counts.values())

            balanced_data = []
            balanced_labels = []

            # Group data by class
            class_data = {}
            for i, label in enumerate(labels):
                if label not in class_data:
                    class_data[label] = []
                class_data[label].append((data[i], label))

            # Undersample each class
            for class_label, samples in class_data.items():
                # Randomly select min_count samples
                selected_samples = random.sample(samples, min_count)

                for sample_data, sample_label in selected_samples:
                    balanced_data.append(sample_data)
                    balanced_labels.append(sample_label)

        else:
            raise DataProcessingError(f"Unknown balancing method: {method}")

        logger.info(f"Balanced dataset using {method}: {len(balanced_data)} samples")
        return balanced_data, balanced_labels

    @staticmethod
    def split_data(
        data: List[List[float]],
        labels: List[int],
        split_ratios: List[float],
        shuffle: bool = True,
    ) -> List[Tuple[List[List[float]], List[int]]]:
        """
        Split data into multiple subsets.

        Args:
            data: Input data
            labels: Input labels
            split_ratios: List of ratios for each split (should sum to 1.0)
            shuffle: Whether to shuffle before splitting

        Returns:
            List of (data_split, labels_split) tuples
        """
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise DataProcessingError("Split ratios must sum to 1.0")

        dataset_size = len(data)
        indices = list(range(dataset_size))

        if shuffle:
            random.shuffle(indices)

        splits = []
        start_idx = 0

        for i, ratio in enumerate(split_ratios):
            if i == len(split_ratios) - 1:
                # Last split gets remaining samples
                end_idx = dataset_size
            else:
                end_idx = start_idx + int(dataset_size * ratio)

            split_indices = indices[start_idx:end_idx]
            split_data = [data[idx] for idx in split_indices]
            split_labels = [labels[idx] for idx in split_indices]

            splits.append((split_data, split_labels))
            start_idx = end_idx

        logger.info(f"Split data into {len(splits)} subsets with ratios {split_ratios}")
        return splits

    @staticmethod
    def compute_data_statistics(data: List[List[float]]) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for dataset.

        Args:
            data: Input data

        Returns:
            Dictionary of statistics
        """
        if not data or not data[0]:
            return {}

        num_samples = len(data)
        num_features = len(data[0])

        # Feature-wise statistics
        feature_stats = []

        for feature_idx in range(num_features):
            feature_values = [sample[feature_idx] for sample in data]

            mean_val = sum(feature_values) / len(feature_values)
            variance = sum((x - mean_val) ** 2 for x in feature_values) / len(
                feature_values
            )
            std_val = math.sqrt(variance)

            feature_stat = {
                "mean": mean_val,
                "std": std_val,
                "variance": variance,
                "min": min(feature_values),
                "max": max(feature_values),
                "range": max(feature_values) - min(feature_values),
            }

            feature_stats.append(feature_stat)

        # Overall statistics
        all_values = [value for sample in data for value in sample]
        overall_mean = sum(all_values) / len(all_values)
        overall_variance = sum((x - overall_mean) ** 2 for x in all_values) / len(
            all_values
        )

        statistics = {
            "num_samples": num_samples,
            "num_features": num_features,
            "feature_statistics": feature_stats,
            "overall_statistics": {
                "mean": overall_mean,
                "std": math.sqrt(overall_variance),
                "variance": overall_variance,
                "min": min(all_values),
                "max": max(all_values),
                "range": max(all_values) - min(all_values),
            },
        }

        return statistics

    @staticmethod
    def analyze_class_distribution(labels: List[int]) -> Dict[str, Any]:
        """
        Analyze class distribution in labels.

        Args:
            labels: List of class labels

        Returns:
            Class distribution analysis
        """
        class_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(class_counts)

        # Calculate class percentages
        class_percentages = {
            class_label: (count / total_samples) * 100
            for class_label, count in class_counts.items()
        }

        # Calculate imbalance metrics
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        # Calculate entropy (measure of class diversity)
        entropy = 0.0
        for count in class_counts.values():
            prob = count / total_samples
            if prob > 0:
                entropy -= prob * math.log2(prob)

        analysis = {
            "num_classes": num_classes,
            "total_samples": total_samples,
            "class_counts": dict(class_counts),
            "class_percentages": class_percentages,
            "imbalance_ratio": imbalance_ratio,
            "entropy": entropy,
            "is_balanced": imbalance_ratio <= 2.0,  # Threshold for "balanced"
            "majority_class": max(class_counts, key=class_counts.get),
            "minority_class": min(class_counts, key=class_counts.get),
        }

        return analysis

    @staticmethod
    def detect_outliers(
        data: List[List[float]], method: str = "iqr", threshold: float = 1.5
    ) -> List[int]:
        """
        Detect outliers in dataset.

        Args:
            data: Input data
            method: Detection method ("iqr", "zscore")
            threshold: Threshold for outlier detection

        Returns:
            List of outlier sample indices
        """
        if not data:
            return []

        outlier_indices = set()

        if method == "iqr":
            # Interquartile Range method
            num_features = len(data[0])

            for feature_idx in range(num_features):
                feature_values = [sample[feature_idx] for sample in data]
                feature_values.sort()

                n = len(feature_values)
                q1_idx = n // 4
                q3_idx = 3 * n // 4

                q1 = feature_values[q1_idx]
                q3 = feature_values[q3_idx]
                iqr = q3 - q1

                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

                # Find outliers for this feature
                for sample_idx, sample in enumerate(data):
                    value = sample[feature_idx]
                    if value < lower_bound or value > upper_bound:
                        outlier_indices.add(sample_idx)

        elif method == "zscore":
            # Z-score method
            num_features = len(data[0])

            for feature_idx in range(num_features):
                feature_values = [sample[feature_idx] for sample in data]

                # Calculate mean and std
                mean_val = sum(feature_values) / len(feature_values)
                variance = sum((x - mean_val) ** 2 for x in feature_values) / len(
                    feature_values
                )
                std_val = math.sqrt(variance)

                if std_val > 0:
                    # Find outliers for this feature
                    for sample_idx, sample in enumerate(data):
                        value = sample[feature_idx]
                        z_score = abs(value - mean_val) / std_val
                        if z_score > threshold:
                            outlier_indices.add(sample_idx)

        else:
            raise DataProcessingError(f"Unknown outlier detection method: {method}")

        outlier_list = sorted(list(outlier_indices))
        logger.info(f"Detected {len(outlier_list)} outliers using {method} method")
        return outlier_list

    @staticmethod
    def create_data_loaders(
        datasets: List[Dict[str, Any]], batch_size: int = 32, shuffle: bool = True
    ) -> List[Any]:
        """
        Create data loaders for federated datasets.

        Args:
            datasets: List of client datasets
            batch_size: Batch size for data loaders
            shuffle: Whether to shuffle data

        Returns:
            List of data loader objects (simplified mock implementation)
        """
        data_loaders = []

        for dataset in datasets:
            data = dataset["data"]
            labels = dataset["labels"]

            # Create simple batched data loader
            loader = {
                "data": data,
                "labels": labels,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_batches": math.ceil(len(data) / batch_size),
                "dataset_size": len(data),
            }

            data_loaders.append(loader)

        logger.info(
            f"Created {len(data_loaders)} data loaders with batch size {batch_size}"
        )
        return data_loaders

    @staticmethod
    def get_batch(
        data_loader: Dict[str, Any], batch_idx: int
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Get a specific batch from data loader.

        Args:
            data_loader: Data loader object
            batch_idx: Batch index

        Returns:
            Tuple of (batch_data, batch_labels)
        """
        batch_size = data_loader["batch_size"]
        data = data_loader["data"]
        labels = data_loader["labels"]

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data))

        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        return batch_data, batch_labels

    @staticmethod
    def validate_data_consistency(datasets: List[Dict[str, Any]]) -> List[str]:
        """
        Validate consistency across federated datasets.

        Args:
            datasets: List of client datasets

        Returns:
            List of validation errors
        """
        errors = []

        if not datasets:
            return ["No datasets provided"]

        # Check basic structure
        first_dataset = datasets[0]
        expected_shape = first_dataset.get("input_shape")
        expected_classes = first_dataset.get("num_classes")

        for i, dataset in enumerate(datasets):
            # Check data-label alignment
            data_len = len(dataset.get("data", []))
            labels_len = len(dataset.get("labels", []))

            if data_len != labels_len:
                errors.append(
                    f"Dataset {i}: data length ({data_len}) != labels length ({labels_len})"
                )

            # Check input shape consistency
            if dataset.get("input_shape") != expected_shape:
                errors.append(f"Dataset {i}: input shape mismatch")

            # Check number of classes
            if dataset.get("num_classes") != expected_classes:
                errors.append(f"Dataset {i}: number of classes mismatch")

            # Check for empty datasets
            if data_len == 0:
                errors.append(f"Dataset {i}: empty dataset")

            # Check feature dimension consistency
            if data_len > 0:
                sample_data = dataset["data"][0]
                if isinstance(sample_data, list):
                    if expected_shape and len(sample_data) != expected_shape[0]:
                        errors.append(f"Dataset {i}: feature dimension mismatch")

        return errors
