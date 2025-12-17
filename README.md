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

THP must be enabled for USM allocation to work performantly. You can check and enable it using the following commands:

```bash
# Check current status
cat /sys/kernel/mm/transparent_hugepage/enabled

# Enable
sudo sh -c 'echo madvise > /sys/kernel/mm/transparent_hugepage/enabled'

# Disable (after tests, if needed)
sudo sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/enabled'
```

## Build Instructions

All tests require custom builds of PyTorch with USM storage support. 

### NVIDIA Jetson (Jetson AGX Orin)

You need to install the [NVIDIA JetPack SDK](https://developer.nvidia.com/embedded/jetpack) including CUDA 12.x.

Then, follow these steps to build PyTorch:

```bash
# Clone the custom PyTorch repo
git clone https://github.com/nagic0/pytorch.git
cd pytorch
git switch dev/usm_storage
git submodule update --init --recursive --progress -j 8

# Create a conda environment
conda create -n usm-test python=3.11 -y
conda activate usm-test

pip install -r ./requirements-build.txt

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
export USE_FLASH_ATTENTION=0  # Disable because of too slow compile on Jetson
export USE_MEM_EFF_ATTENTION=0
# export MAX_JOBS=5  # Adjust based on your system memory if compile with flash attention

# Build and install
python setup.py install
```

### Intel iGPU Platforms (Arrow Lake)

You need to install the [Intel oneAPI Base Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).

Then, follow these steps to build PyTorch with special [torch-xpu-ops](https://github.com/nagic0/torch-xpu-ops/tree/dev/torch_usm_l0).

```bash
git clone https://github.com/nagic0/pytorch.git
cd pytorch
git switch dev/usm_storage_intel
git submodule update --init --recursive --progress -j 8

# Create a conda environment
conda create -n usm-test python=3.11 -y
conda activate usm-test

pip install -r ./requirements-build.txt

source /opt/intel/oneapi/setvars.sh

export BUILD_TEST=0
export INSTALL_TEST=0
export PYTORCH_QNNPACK_BUILD_TESTS=0
export PTHREADPOOL_BUILD_TESTS=0
export XNNPACK_BUILD_TESTS=0
export DNNL_BUILD_TESTS=0
export USE_CUDA=0
export USE_ROCM=0
export USE_XCCL=0
export USE_DISTRIBUTED=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export USE_XNNPACK=0
export REL_WITH_DEB_INFO=1
export TORCH_XPU_ARCH_LIST="arl-h"

python setup.py install
```