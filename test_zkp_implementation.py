#!/usr/bin/env python
"""
Test script to validate ZKP implementation and ensure everything works.

This script tests:
1. ZKP parameter generation
2. ZKP commitment and verification
3. Benchmark tracking
4. Client/server integration (imports and initialization)
"""

import sys
import os
import tempfile
import shutil
import numpy as np

print("="*60)
print("ZKP Implementation Validation Test")
print("="*60)

# Test 1: Import all new modules
print("\n[Test 1] Testing imports...")

# First, check if we have required dependencies
try:
    import torch
    print("✓ PyTorch available")
except ImportError:
    print("✗ PyTorch not found - please install: conda install pytorch")
    sys.exit(1)

try:
    import psutil
    print("✓ psutil available")
except ImportError:
    print("⚠ psutil not found - installing from requirements.txt recommended")
    print("  Some benchmark features may not work")

try:
    from core.zkp import (
        ZKPContext, ZKPLayer, create_zkp_context, 
        zkp_commit_model, aggregate_zkp_layers, verify_zkp_layers,
        read_zkp_params, write_zkp_params
    )
    print("✓ ZKP module imports successful")
except Exception as e:
    print(f"✗ ZKP import failed: {e}")
    sys.exit(1)

try:
    from core.benchmark import (
        BenchmarkMetrics, BenchmarkTimer, 
        init_benchmark, get_benchmark,
        get_memory_usage_mb, estimate_params_size
    )
    print("✓ Benchmark module imports successful")
except Exception as e:
    print(f"✗ Benchmark import failed: {e}")
    sys.exit(1)

try:
    from core.common import get_parameters2, set_parameters
    from core.model_builder import Net
    import torch
    print("✓ Core modules imports successful")
except Exception as e:
    print(f"✗ Core import failed: {e}")
    sys.exit(1)

# Test 2: ZKP Context Creation
print("\n[Test 2] Testing ZKP context creation...")
try:
    # Create a smaller context for testing (512-bit for speed)
    # Note: Not secure, just for testing
    print("  Creating ZKP context (this may take a moment)...")
    context = create_zkp_context(bit_length=2048)
    print(f"✓ ZKP context created")
    print(f"  - Prime p: {context.p.bit_length()} bits")
    print(f"  - Group order q: {context.q.bit_length()} bits")
    print(f"  - Generator g: {context.g}")
    print(f"  - Generator h: {context.h}")
except Exception as e:
    print(f"✗ ZKP context creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Pedersen Commitments
print("\n[Test 3] Testing Pedersen commitments...")
try:
    test_value = 42.12345
    commitment, randomness = context.commit(test_value)
    print(f"✓ Created commitment for value {test_value}")
    print(f"  - Commitment: {commitment}")
    print(f"  - Randomness: {randomness}")
    
    # Verify commitment
    is_valid = context.verify_commitment(test_value, commitment, randomness)
    if is_valid:
        print("✓ Commitment verification successful")
    else:
        print("✗ Commitment verification failed")
        sys.exit(1)
    
    # Test invalid verification
    is_invalid = context.verify_commitment(99.99, commitment, randomness)
    if not is_invalid:
        print("✓ Invalid commitment correctly rejected")
    else:
        print("✗ Invalid commitment incorrectly accepted")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Commitment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: ZKP Layer Creation
print("\n[Test 4] Testing ZKP layer creation...")
try:
    # Create a small weight array
    weights = np.random.randn(3, 3).astype(np.float32)
    
    # Create ZKP layer without context (baseline)
    layer_plain = ZKPLayer("test_layer", weights, context=None)
    print(f"✓ Created plain layer: {layer_plain.name}")
    
    # Create ZKP layer with commitments
    print("  Creating layer with commitments...")
    layer_zkp = ZKPLayer("test_layer_zkp", weights, context=context)
    print(f"✓ Created ZKP layer: {layer_zkp.name}")
    print(f"  - Number of commitments: {len(layer_zkp.commitments) if layer_zkp.commitments else 0}")
    
    # Verify the layer
    if layer_zkp.verify():
        print("✓ ZKP layer verification successful")
    else:
        print("✗ ZKP layer verification failed")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ ZKP layer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Model Commitment
print("\n[Test 5] Testing model commitment...")
try:
    # Create a small model
    model = Net(num_classes=10)
    state_dict = model.state_dict()
    
    print(f"  Model has {len(state_dict)} layers")
    
    # Create commitments for model
    print("  Creating ZKP commitments for model (this may take a moment)...")
    zkp_layers = zkp_commit_model(state_dict, context, protected_layers=['fc3.weight'])
    
    print(f"✓ Created {len(zkp_layers)} ZKP layers")
    
    # Check which layers are protected
    protected_count = sum(1 for layer in zkp_layers if layer.context is not None)
    print(f"  - Protected layers: {protected_count}")
    print(f"  - Unprotected layers: {len(zkp_layers) - protected_count}")
    
    # Verify all layers
    if verify_zkp_layers(zkp_layers):
        print("✓ All ZKP layers verified successfully")
    else:
        print("✗ ZKP layer verification failed")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Model commitment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: ZKP Aggregation
print("\n[Test 6] Testing ZKP layer aggregation...")
try:
    # Create multiple sets of ZKP layers (simulating multiple clients)
    num_clients = 3
    all_client_layers = []
    
    for client_id in range(num_clients):
        model = Net(num_classes=10)
        state_dict = model.state_dict()
        zkp_layers = zkp_commit_model(state_dict, context, protected_layers=['fc3.weight'])
        all_client_layers.append((zkp_layers, 100))  # 100 examples per client
    
    print(f"  Created ZKP layers for {num_clients} clients")
    
    # Aggregate
    print("  Aggregating ZKP layers...")
    aggregated = aggregate_zkp_layers(all_client_layers)
    
    print(f"✓ Aggregated {len(aggregated)} layers")
    print(f"  - Each aggregated layer shape: {aggregated[0].get_weights().shape}")
    
except Exception as e:
    print(f"✗ Aggregation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Benchmark Tracking
print("\n[Test 7] Testing benchmark tracking...")
try:
    # Initialize benchmark
    metrics = init_benchmark("zkp", num_clients=3, rounds=2)
    print("✓ Benchmark initialized")
    
    # Test timing
    import time
    with BenchmarkTimer(metrics, 'client_fit'):
        time.sleep(0.1)
    print("✓ Timer context manager works")
    
    # Add various metrics
    metrics.add_client_memory(512.5)
    metrics.add_upload_size(1024 * 1024)  # 1 MB
    metrics.add_proof_generation(0.5)
    
    print("✓ Metrics added successfully")
    
    # Get summary
    summary = metrics.summary()
    print(f"  - Mode: {summary['mode']}")
    print(f"  - Client fit time: {summary['timing']['client_fit']['total']:.3f}s")
    print(f"  - Upload size: {summary['communication_bytes']['upload']['total'] / 1024:.1f} KB")
    
    # Test save
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    metrics.save(temp_file)
    print(f"✓ Benchmark saved to {temp_file}")
    
    # Clean up
    os.remove(temp_file)
    
except Exception as e:
    print(f"✗ Benchmark test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Parameter Serialization
print("\n[Test 8] Testing ZKP parameter serialization...")
try:
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    param_file = os.path.join(temp_dir, "test_zkp_params.pkl")
    
    # Save parameters
    write_zkp_params(param_file, context)
    print(f"✓ ZKP parameters saved to {param_file}")
    
    # Load parameters
    loaded_context = read_zkp_params(param_file)
    print("✓ ZKP parameters loaded")
    
    # Verify loaded context works
    test_val = 123.456
    c1, r1 = context.commit(test_val)
    c2, r2 = loaded_context.commit(test_val)
    
    # They should produce different commitments (different randomness)
    # but both should be valid
    if loaded_context.verify_commitment(test_val, c1, r1):
        print("✓ Loaded context can verify commitments")
    else:
        print("✗ Loaded context verification failed")
        sys.exit(1)
    
    # Clean up
    shutil.rmtree(temp_dir)
    
except Exception as e:
    print(f"✗ Serialization test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Integration with get/set parameters
print("\n[Test 9] Testing integration with get/set parameters...")
try:
    model = Net(num_classes=10)
    
    # Test baseline (no crypto)
    params_baseline = get_parameters2(model, context_client=None, zkp_context=None)
    print(f"✓ Baseline parameters: {len(params_baseline)} arrays")
    
    # Test ZKP
    params_zkp = get_parameters2(model, context_client=None, zkp_context=context)
    print(f"✓ ZKP parameters: {len(params_zkp)} ZKPLayers")
    
    # Test set parameters with ZKP
    model2 = Net(num_classes=10)
    set_parameters(model2, params_zkp, context_client=None, zkp_context=context)
    print("✓ Set parameters with ZKP successful")
    
    # Verify models have same weights
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), model2.state_dict().items()):
        if not torch.allclose(p1, p2, rtol=1e-4):
            print(f"✗ Parameter mismatch in layer {n1}")
            sys.exit(1)
    print("✓ Parameters match after set")
    
except Exception as e:
    print(f"✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Memory and size estimation
print("\n[Test 10] Testing utility functions...")
try:
    # Test memory usage
    mem_usage = get_memory_usage_mb()
    print(f"✓ Current memory usage: {mem_usage:.2f} MB")
    
    # Test parameter size estimation
    model = Net(num_classes=10)
    params = [p.detach().cpu().numpy() for p in model.parameters()]
    size_bytes = estimate_params_size(params)
    print(f"✓ Model parameter size: {size_bytes / 1024:.2f} KB")
    
except Exception as e:
    print(f"✗ Utility test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("\n" + "="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
print("\nSummary:")
print("✓ ZKP module imports")
print("✓ ZKP context creation (2048-bit)")
print("✓ Pedersen commitments and verification")
print("✓ ZKP layer creation and verification")
print("✓ Model commitment (selective layer protection)")
print("✓ Multi-client ZKP aggregation")
print("✓ Benchmark tracking and metrics")
print("✓ Parameter serialization/deserialization")
print("✓ Integration with get/set parameters")
print("✓ Utility functions (memory, size estimation)")
print("\n" + "="*60)
print("The ZKP implementation is ready to use!")
print("="*60)

print("\nNext steps:")
print("1. Generate ZKP parameters: python create_zkp_params.py")
print("2. Run with ZKP: python simulation.py simulation --zkp --zkp_params zkp_params.pkl ...")
print("3. Compare methods: python compare_methods.py --modes baseline,zkp ...")
