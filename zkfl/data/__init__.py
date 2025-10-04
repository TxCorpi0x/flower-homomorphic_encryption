"""Data handling components for ZKFL."""

from .data_setup import DataSetup
from .data_utils import DataUtils


# Convenience functions
def get_dataset(dataset_name: str = "cifar10", **kwargs):
    """Get a dataset by name."""
    setup = DataSetup()
    return setup.load_dataset(dataset_name, **kwargs)


# Alias for compatibility
DataLoader = DataUtils

__all__ = ["DataSetup", "DataUtils", "get_dataset", "DataLoader"]
