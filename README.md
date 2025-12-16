# USM Loading Tests

This directory contains a comprehensive test suite for evaluating PyTorch model loading performance using Unified Shared Memory (USM). Tests are organized in two categories:

1. **Storage Tests** (`1-storage/`) — Low-level tests for loading binary weight files directly.
2. **LLM Tests** (`2-llm-safetensors/`) — High-level tests for loading LLM models via Safetensors and Transformers.

Both test suites compare standard copy-based loading against USM-backed storage approaches to measure performance improvements.

## System Requirements

- CUDA-capable USM device (e.g., NVIDIA Jetson, or similar GPU with USM support)
- Linux system with transparent hugepage support
- Python 3.8+ environment with conda

## Prerequisites

### Enable Transparent Hugepage (THP)

THP must be enabled for USM allocation to work properly:

```bash
# Check current status
cat /sys/kernel/mm/transparent_hugepage/enabled

# Enable (recommended)
sudo sh -c 'echo madvise > /sys/kernel/mm/transparent_hugepage/enabled'

# Disable (if needed, not recommended)
sudo sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/enabled'
```

## Build Instructions

All tests require custom builds of PyTorch with USM storage support. Below is an example for NVIDIA Jetson AGX Orin.

### Build Custom PyTorch

```bash
# Clone the custom PyTorch repo
git clone https://github.com/nagic0/pytorch.git
cd pytorch

# Check out the USM storage branch
git switch dev/usm_storage

# Create a conda environment
conda create -n usm-test python=3.11 -y
conda activate usm-test

# Build configuration
export BUILD_TEST=0
export INSTALL_TEST=0
export PYTORCH_QNNPACK_BUILD_TESTS=0
export PTHREADPOOL_BUILD_TESTS=0
export XNNPACK_BUILD_TESTS=0
export DNNL_BUILD_TESTS=0
export USE_NCCL=0
export USE_DISTRIBUTED=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export USE_CUFILE=0
export TORCH_CUDA_ARCH_LIST="8.7"  # Set according to your GPU
export REL_WITH_DEB_INFO=1
export USE_PRIORITIZED_TEXT_FOR_LD=1
export MAX_JOBS=5  # Adjust based on your system memory

# Build and install
python setup.py install
```
