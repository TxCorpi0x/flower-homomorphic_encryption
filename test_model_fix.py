#!/usr/bin/env python3
"""
Quick test script to verify the model building fix.
"""

import sys
import os

sys.path.append("/Users/scorpio/github.com/TxCorpi0x/flower-homomorphic_encryption")

from zkfl.models.model_builder import ModelBuilder
import torch.nn as nn


def test_model_building():
    """Test that ModelBuilder returns proper PyTorch nn.Module instances."""
    print("🧪 Testing ZKFL ModelBuilder...")

    builder = ModelBuilder()

    # Test simple CNN
    print("Testing SimpleCNN...")
    model = builder.build_model(
        model_type="simple_cnn",
        input_shape=(3, 32, 32),  # CIFAR-10 shape
        num_classes=10,
        hidden_units=128,
        dropout_rate=0.25,
    )

    # Verify it's a PyTorch module
    assert isinstance(model, nn.Module), f"Expected nn.Module, got {type(model)}"
    print(f"✅ SimpleCNN: {type(model)} - SUCCESS")

    # Test MLP
    print("Testing SimpleMLP...")
    mlp_model = builder.build_model(
        model_type="mlp",
        input_shape=(28, 28),  # MNIST-like shape
        num_classes=10,
        hidden_units=128,
    )

    assert isinstance(
        mlp_model, nn.Module
    ), f"Expected nn.Module, got {type(mlp_model)}"
    print(f"✅ SimpleMLP: {type(mlp_model)} - SUCCESS")

    print("🎉 All model building tests passed!")
    print(f"📊 Supported models: {builder.get_supported_models()}")

    return True


if __name__ == "__main__":
    try:
        test_model_building()
        print("\n✅ MODEL FIX VERIFICATION: SUCCESS")
        print("The production server should now be able to build models correctly!")
    except Exception as e:
        print(f"\n❌ MODEL FIX VERIFICATION: FAILED")
        print(f"Error: {e}")
        sys.exit(1)
