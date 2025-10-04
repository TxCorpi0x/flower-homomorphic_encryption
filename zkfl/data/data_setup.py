"""
Data setup utilities for ZKFL federated learning.

This module provides functions for data loading, preprocessing,
and federated data partitioning for federated learning scenarios.
"""

import logging
import random
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

from ..core.exceptions import DataProcessingError, ConfigurationError

logger = logging.getLogger(__name__)


class DataSetup:
    """Data setup and partitioning for federated learning."""

    def __init__(self):
        """Initialize data setup utility."""
        self.supported_datasets = {
            "cifar10": self._load_cifar10,
            "mnist": self._load_mnist,
            "fashion_mnist": self._load_fashion_mnist,
            "synthetic": self._generate_synthetic_data,
            "custom": self._load_custom_data,
        }
        logger.info("DataSetup initialized")

    def create_federated_dataset(
        self,
        dataset_name: str,
        num_clients: int,
        partition_method: str = "iid",
        alpha: float = 0.5,
        min_samples_per_client: int = 10,
        **kwargs,
    ) -> Tuple[List[Any], Any]:
        """
        Create federated dataset partitioned across clients.

        Args:
            dataset_name: Name of dataset to load
            num_clients: Number of federated clients
            partition_method: Method for partitioning ("iid", "non_iid", "dirichlet")
            alpha: Concentration parameter for Dirichlet distribution
            min_samples_per_client: Minimum samples per client
            **kwargs: Additional dataset-specific parameters

        Returns:
            Tuple of (client_datasets, test_dataset)
        """
        try:
            logger.info(
                f"Creating federated dataset: {dataset_name} for {num_clients} clients"
            )

            # Load base dataset
            if dataset_name not in self.supported_datasets:
                raise DataProcessingError(f"Unsupported dataset: {dataset_name}")

            train_data, test_data = self.supported_datasets[dataset_name](**kwargs)

            # Partition data among clients
            client_datasets = self._partition_data(
                train_data, num_clients, partition_method, alpha, min_samples_per_client
            )

            logger.info(
                f"Created federated dataset with {len(client_datasets)} client partitions"
            )
            return client_datasets, test_data

        except Exception as e:
            logger.error(f"Failed to create federated dataset: {e}")
            raise DataProcessingError(f"Federated dataset creation failed: {e}")

    def _load_cifar10(self, **kwargs) -> Tuple[Any, Any]:
        """
        Load CIFAR-10 dataset.

        Returns:
            Tuple of (train_data, test_data)
        """
        try:
            # In practice, this would use torchvision or tensorflow datasets
            # For simulation, create mock data structure

            num_train_samples = kwargs.get("num_train_samples", 50000)
            num_test_samples = kwargs.get("num_test_samples", 10000)

            train_data = {
                "data": self._generate_image_data((num_train_samples, 32, 32, 3)),
                "labels": self._generate_labels(num_train_samples, 10),
                "dataset_name": "cifar10",
                "num_classes": 10,
                "input_shape": (32, 32, 3),
            }

            test_data = {
                "data": self._generate_image_data((num_test_samples, 32, 32, 3)),
                "labels": self._generate_labels(num_test_samples, 10),
                "dataset_name": "cifar10",
                "num_classes": 10,
                "input_shape": (32, 32, 3),
            }

            logger.info(
                f"Loaded CIFAR-10: {num_train_samples} train, {num_test_samples} test"
            )
            return train_data, test_data

        except Exception as e:
            logger.error(f"Failed to load CIFAR-10: {e}")
            raise DataProcessingError(f"CIFAR-10 loading failed: {e}")

    def _load_mnist(self, **kwargs) -> Tuple[Any, Any]:
        """
        Load MNIST dataset.

        Returns:
            Tuple of (train_data, test_data)
        """
        try:
            num_train_samples = kwargs.get("num_train_samples", 60000)
            num_test_samples = kwargs.get("num_test_samples", 10000)

            train_data = {
                "data": self._generate_image_data((num_train_samples, 28, 28, 1)),
                "labels": self._generate_labels(num_train_samples, 10),
                "dataset_name": "mnist",
                "num_classes": 10,
                "input_shape": (28, 28, 1),
            }

            test_data = {
                "data": self._generate_image_data((num_test_samples, 28, 28, 1)),
                "labels": self._generate_labels(num_test_samples, 10),
                "dataset_name": "mnist",
                "num_classes": 10,
                "input_shape": (28, 28, 1),
            }

            logger.info(
                f"Loaded MNIST: {num_train_samples} train, {num_test_samples} test"
            )
            return train_data, test_data

        except Exception as e:
            logger.error(f"Failed to load MNIST: {e}")
            raise DataProcessingError(f"MNIST loading failed: {e}")

    def _load_fashion_mnist(self, **kwargs) -> Tuple[Any, Any]:
        """
        Load Fashion-MNIST dataset.

        Returns:
            Tuple of (train_data, test_data)
        """
        try:
            num_train_samples = kwargs.get("num_train_samples", 60000)
            num_test_samples = kwargs.get("num_test_samples", 10000)

            train_data = {
                "data": self._generate_image_data((num_train_samples, 28, 28, 1)),
                "labels": self._generate_labels(num_train_samples, 10),
                "dataset_name": "fashion_mnist",
                "num_classes": 10,
                "input_shape": (28, 28, 1),
            }

            test_data = {
                "data": self._generate_image_data((num_test_samples, 28, 28, 1)),
                "labels": self._generate_labels(num_test_samples, 10),
                "dataset_name": "fashion_mnist",
                "num_classes": 10,
                "input_shape": (28, 28, 1),
            }

            logger.info(
                f"Loaded Fashion-MNIST: {num_train_samples} train, {num_test_samples} test"
            )
            return train_data, test_data

        except Exception as e:
            logger.error(f"Failed to load Fashion-MNIST: {e}")
            raise DataProcessingError(f"Fashion-MNIST loading failed: {e}")

    def _generate_synthetic_data(self, **kwargs) -> Tuple[Any, Any]:
        """
        Generate synthetic dataset.

        Returns:
            Tuple of (train_data, test_data)
        """
        try:
            num_train_samples = kwargs.get("num_train_samples", 1000)
            num_test_samples = kwargs.get("num_test_samples", 200)
            num_features = kwargs.get("num_features", 100)
            num_classes = kwargs.get("num_classes", 2)

            train_data = {
                "data": self._generate_tabular_data((num_train_samples, num_features)),
                "labels": self._generate_labels(num_train_samples, num_classes),
                "dataset_name": "synthetic",
                "num_classes": num_classes,
                "input_shape": (num_features,),
            }

            test_data = {
                "data": self._generate_tabular_data((num_test_samples, num_features)),
                "labels": self._generate_labels(num_test_samples, num_classes),
                "dataset_name": "synthetic",
                "num_classes": num_classes,
                "input_shape": (num_features,),
            }

            logger.info(
                f"Generated synthetic data: {num_train_samples} train, {num_test_samples} test"
            )
            return train_data, test_data

        except Exception as e:
            logger.error(f"Failed to generate synthetic data: {e}")
            raise DataProcessingError(f"Synthetic data generation failed: {e}")

    def _load_custom_data(self, **kwargs) -> Tuple[Any, Any]:
        """
        Load custom dataset from file paths.

        Returns:
            Tuple of (train_data, test_data)
        """
        try:
            train_path = kwargs.get("train_path")
            test_path = kwargs.get("test_path")

            if not train_path or not test_path:
                raise DataProcessingError(
                    "train_path and test_path must be provided for custom data"
                )

            # In practice, this would load from actual files
            # For simulation, create mock data

            train_data = {
                "data": self._generate_tabular_data((100, 50)),
                "labels": self._generate_labels(100, 2),
                "dataset_name": "custom",
                "num_classes": 2,
                "input_shape": (50,),
                "source_path": train_path,
            }

            test_data = {
                "data": self._generate_tabular_data((20, 50)),
                "labels": self._generate_labels(20, 2),
                "dataset_name": "custom",
                "num_classes": 2,
                "input_shape": (50,),
                "source_path": test_path,
            }

            logger.info(f"Loaded custom data from {train_path} and {test_path}")
            return train_data, test_data

        except Exception as e:
            logger.error(f"Failed to load custom data: {e}")
            raise DataProcessingError(f"Custom data loading failed: {e}")

    def _partition_data(
        self,
        data: Dict[str, Any],
        num_clients: int,
        method: str,
        alpha: float,
        min_samples_per_client: int,
    ) -> List[Dict[str, Any]]:
        """
        Partition data among federated clients.

        Args:
            data: Dataset to partition
            num_clients: Number of clients
            method: Partitioning method
            alpha: Concentration parameter for non-IID partitioning
            min_samples_per_client: Minimum samples per client

        Returns:
            List of client datasets
        """
        dataset_size = len(data["labels"])

        if dataset_size < num_clients * min_samples_per_client:
            raise DataProcessingError(
                f"Dataset too small: {dataset_size} < {num_clients * min_samples_per_client}"
            )

        if method == "iid":
            return self._partition_iid(data, num_clients)
        elif method == "non_iid":
            return self._partition_non_iid(data, num_clients, alpha)
        elif method == "dirichlet":
            return self._partition_dirichlet(data, num_clients, alpha)
        else:
            raise DataProcessingError(f"Unknown partitioning method: {method}")

    def _partition_iid(
        self, data: Dict[str, Any], num_clients: int
    ) -> List[Dict[str, Any]]:
        """Partition data in IID manner."""
        dataset_size = len(data["labels"])
        indices = list(range(dataset_size))
        random.shuffle(indices)

        client_datasets = []
        samples_per_client = dataset_size // num_clients

        for i in range(num_clients):
            start_idx = i * samples_per_client
            if i == num_clients - 1:
                # Last client gets remaining samples
                end_idx = dataset_size
            else:
                end_idx = (i + 1) * samples_per_client

            client_indices = indices[start_idx:end_idx]

            client_data = {
                "data": self._select_samples(data["data"], client_indices),
                "labels": [data["labels"][idx] for idx in client_indices],
                "client_id": i,
                "dataset_name": data["dataset_name"],
                "num_classes": data["num_classes"],
                "input_shape": data["input_shape"],
                "partition_method": "iid",
            }

            client_datasets.append(client_data)

        return client_datasets

    def _partition_non_iid(
        self, data: Dict[str, Any], num_clients: int, alpha: float
    ) -> List[Dict[str, Any]]:
        """Partition data in non-IID manner by class."""
        num_classes = data["num_classes"]
        labels = data["labels"]

        # Group samples by class
        class_indices = {}
        for idx, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Distribute classes among clients
        client_datasets = []
        classes_per_client = max(1, num_classes // num_clients)

        available_classes = list(range(num_classes))
        random.shuffle(available_classes)

        for i in range(num_clients):
            # Select classes for this client
            start_class = (i * classes_per_client) % num_classes
            client_classes = []

            for j in range(classes_per_client):
                class_idx = (start_class + j) % num_classes
                client_classes.append(class_idx)

            # Collect samples from selected classes
            client_indices = []
            for class_label in client_classes:
                if class_label in class_indices:
                    class_samples = class_indices[class_label]
                    # Take a portion of samples from this class
                    num_samples = len(class_samples) // max(
                        1, (num_clients // classes_per_client)
                    )
                    client_indices.extend(class_samples[:num_samples])

            if client_indices:
                client_data = {
                    "data": self._select_samples(data["data"], client_indices),
                    "labels": [data["labels"][idx] for idx in client_indices],
                    "client_id": i,
                    "dataset_name": data["dataset_name"],
                    "num_classes": data["num_classes"],
                    "input_shape": data["input_shape"],
                    "partition_method": "non_iid",
                    "client_classes": client_classes,
                }

                client_datasets.append(client_data)

        return client_datasets

    def _partition_dirichlet(
        self, data: Dict[str, Any], num_clients: int, alpha: float
    ) -> List[Dict[str, Any]]:
        """Partition data using Dirichlet distribution."""
        # Simplified Dirichlet partitioning
        # In practice, would use numpy.random.dirichlet

        num_classes = data["num_classes"]
        labels = data["labels"]

        # Group samples by class
        class_indices = {}
        for idx, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        client_datasets = []

        for i in range(num_clients):
            client_indices = []

            # Generate class proportions (simplified)
            # In practice, use numpy.random.dirichlet([alpha] * num_classes)
            proportions = [random.random() for _ in range(num_classes)]
            total = sum(proportions)
            proportions = [p / total for p in proportions]

            # Sample from each class based on proportions
            target_size = len(labels) // num_clients

            for class_label, proportion in enumerate(proportions):
                if class_label in class_indices:
                    class_samples = class_indices[class_label]
                    num_samples = int(target_size * proportion)
                    num_samples = min(num_samples, len(class_samples))

                    selected_samples = random.sample(class_samples, num_samples)
                    client_indices.extend(selected_samples)

            if client_indices:
                client_data = {
                    "data": self._select_samples(data["data"], client_indices),
                    "labels": [data["labels"][idx] for idx in client_indices],
                    "client_id": i,
                    "dataset_name": data["dataset_name"],
                    "num_classes": data["num_classes"],
                    "input_shape": data["input_shape"],
                    "partition_method": "dirichlet",
                    "alpha": alpha,
                }

                client_datasets.append(client_data)

        return client_datasets

    def _generate_image_data(self, shape: Tuple[int, ...]) -> List[List[float]]:
        """Generate mock image data."""
        total_elements = 1
        for dim in shape:
            total_elements *= dim

        # Generate random pixel values (0-255 normalized to 0-1)
        data = []
        for _ in range(shape[0]):  # Number of samples
            sample = [random.random() for _ in range(total_elements // shape[0])]
            data.append(sample)

        return data

    def _generate_tabular_data(self, shape: Tuple[int, int]) -> List[List[float]]:
        """Generate mock tabular data."""
        data = []
        for _ in range(shape[0]):  # Number of samples
            sample = [random.gauss(0, 1) for _ in range(shape[1])]  # Gaussian features
            data.append(sample)

        return data

    def _generate_labels(self, num_samples: int, num_classes: int) -> List[int]:
        """Generate random labels."""
        return [random.randint(0, num_classes - 1) for _ in range(num_samples)]

    def _select_samples(self, data: List[Any], indices: List[int]) -> List[Any]:
        """Select samples from data using indices."""
        return [data[idx] for idx in indices]

    def get_dataset_statistics(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about federated datasets.

        Args:
            datasets: List of client datasets

        Returns:
            Dataset statistics
        """
        if not datasets:
            return {}

        total_samples = sum(len(dataset["labels"]) for dataset in datasets)
        samples_per_client = [len(dataset["labels"]) for dataset in datasets]

        # Class distribution analysis
        overall_class_counts = {}
        client_class_distributions = []

        for dataset in datasets:
            class_counts = {}
            for label in dataset["labels"]:
                class_counts[label] = class_counts.get(label, 0) + 1
                overall_class_counts[label] = overall_class_counts.get(label, 0) + 1
            client_class_distributions.append(class_counts)

        statistics = {
            "num_clients": len(datasets),
            "total_samples": total_samples,
            "samples_per_client": {
                "min": min(samples_per_client),
                "max": max(samples_per_client),
                "mean": total_samples / len(datasets),
                "distribution": samples_per_client,
            },
            "class_distribution": {
                "overall": overall_class_counts,
                "per_client": client_class_distributions,
            },
            "dataset_info": {
                "name": datasets[0]["dataset_name"],
                "num_classes": datasets[0]["num_classes"],
                "input_shape": datasets[0]["input_shape"],
            },
        }

        return statistics

    def validate_federated_datasets(self, datasets: List[Dict[str, Any]]) -> List[str]:
        """
        Validate federated datasets for consistency.

        Args:
            datasets: List of client datasets

        Returns:
            List of validation errors
        """
        errors = []

        if not datasets:
            errors.append("No datasets provided")
            return errors

        # Check consistency across clients
        first_dataset = datasets[0]
        expected_keys = {"data", "labels", "dataset_name", "num_classes", "input_shape"}

        for i, dataset in enumerate(datasets):
            # Check required keys
            missing_keys = expected_keys - set(dataset.keys())
            if missing_keys:
                errors.append(f"Client {i} missing keys: {missing_keys}")

            # Check consistency with first dataset
            if dataset.get("dataset_name") != first_dataset.get("dataset_name"):
                errors.append(f"Client {i} dataset name mismatch")

            if dataset.get("num_classes") != first_dataset.get("num_classes"):
                errors.append(f"Client {i} num_classes mismatch")

            if dataset.get("input_shape") != first_dataset.get("input_shape"):
                errors.append(f"Client {i} input_shape mismatch")

            # Check data-label alignment
            if len(dataset.get("data", [])) != len(dataset.get("labels", [])):
                errors.append(f"Client {i} data-label length mismatch")

        return errors


# Convenience function for the example script
def create_federated_datasets(
    num_clients: int = 3,
    dataset_name: str = "cifar10",
    partition_type: str = "iid",
    **kwargs,
):
    """
    Convenience function to create federated datasets.

    Args:
        num_clients: Number of federated clients
        dataset_name: Name of the dataset
        partition_type: How to partition data among clients
        **kwargs: Additional arguments

    Returns:
        List of federated datasets
    """
    setup = DataSetup()
    return setup.create_federated_dataset(
        num_clients=num_clients,
        dataset_name=dataset_name,
        partition_type=partition_type,
        **kwargs,
    )
