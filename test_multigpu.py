#!/usr/bin/env python3
"""
Multi-GPU Setup Test Script
Tests GPU detection and multi-GPU configuration
"""

import sys
import torch

def test_cuda_availability():
    """Test if CUDA is available"""
    print("="*60)
    print("CUDA Availability Test")
    print("="*60)

    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
    else:
        print("❌ CUDA is NOT available")
        print("   Please install CUDA and PyTorch with CUDA support")
        return False

    return True

def test_gpu_detection():
    """Detect and list all available GPUs"""
    print("\n" + "="*60)
    print("GPU Detection Test")
    print("="*60)

    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs detected: {gpu_count}")

    if gpu_count == 0:
        print("❌ No GPUs detected")
        return False

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"\n  GPU {i}:")
        print(f"    Name: {gpu_name}")
        print(f"    Memory: {gpu_memory:.2f} GB")
        print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")

    return True

def test_multi_gpu_config():
    """Test multi-GPU configuration strings"""
    print("\n" + "="*60)
    print("Multi-GPU Configuration Test")
    print("="*60)

    gpu_count = torch.cuda.device_count()

    # Test configurations
    test_configs = [
        ("auto", "Auto-detect all GPUs"),
        ("0", "Single GPU 0"),
        ("cpu", "CPU only"),
    ]

    if gpu_count >= 2:
        test_configs.append(("0,1", "GPUs 0 and 1"))

    if gpu_count >= 4:
        test_configs.append(("0,1,2,3", "All 4 GPUs"))

    print(f"\nAvailable configurations for your system ({gpu_count} GPU(s)):\n")

    for config, description in test_configs:
        print(f"  ✅ '{config}' - {description}")

    return True

def test_device_parsing():
    """Test device string parsing logic"""
    print("\n" + "="*60)
    print("Device String Parsing Test")
    print("="*60)

    # Import the parsing function from process_and_train.py
    try:
        from process_and_train import parse_device_string

        test_cases = [
            "auto",
            "0",
            "0,1,2,3",
            "cpu"
        ]

        print("\nTesting device string parsing:\n")
        for test_case in test_cases:
            try:
                result = parse_device_string(test_case)
                print(f"  Input: '{test_case}' → Output: '{result}' ✅")
            except Exception as e:
                print(f"  Input: '{test_case}' → Error: {e} ❌")

        return True

    except ImportError as e:
        print(f"❌ Could not import parse_device_string: {e}")
        return False

def test_tensor_operations():
    """Test basic tensor operations on GPU"""
    print("\n" + "="*60)
    print("GPU Tensor Operations Test")
    print("="*60)

    if not torch.cuda.is_available():
        print("⏭️  Skipping (no CUDA)")
        return True

    try:
        # Create a tensor on GPU
        device = torch.device("cuda:0")
        tensor = torch.randn(1000, 1000).to(device)
        result = tensor @ tensor  # Matrix multiplication

        print(f"✅ Tensor operations on GPU 0 successful")
        print(f"   Tensor shape: {result.shape}")
        print(f"   Device: {result.device}")

        return True

    except Exception as e:
        print(f"❌ GPU tensor operations failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "MULTI-GPU SYSTEM TEST" + " "*22 + "║")
    print("╚" + "="*58 + "╝")
    print()

    results = []

    # Run tests
    results.append(("CUDA Availability", test_cuda_availability()))
    results.append(("GPU Detection", test_gpu_detection()))
    results.append(("Multi-GPU Configuration", test_multi_gpu_config()))
    results.append(("Device String Parsing", test_device_parsing()))
    results.append(("GPU Tensor Operations", test_tensor_operations()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60 + "\n")

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:<30} {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! Your system is ready for multi-GPU training.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("="*60 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
