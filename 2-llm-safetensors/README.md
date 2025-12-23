# USM Loading Test on LLM

This directory contains tests and benchmarks to evaluate loading LLM model weights (via Safetensors) into PyTorch using two methods:

- Standard copy-based file->CPU->GPU path.
- USM-backed storage (Unified Shared Memory) path that aims to avoid an explicit host copy.

The tests are intended for systems with custom PyTorch, Safetensors, and Transformers builds that expose USM-related APIs.

## Repository Layout

- `correctness.py` — Verifies model equivalence between the two loading methods.
- `performance.py` — Runs an automated micro-benchmark for model loading across multiple methods.
- `performance_worker.py` — Worker script invoked by `performance.py` for a single timed run.
- `chat.py` — Interactive chat demo to test model inference.
- `chat_performance.py` — Measures loading and inference latency with and without USM. 

## Prerequisites

- Python 3.8+ with `numpy` and `accelerate` installed.
- Custom PyTorch build with USM storage support (see [main README](../README.md)).
- Custom Safetensors build with USM storage support.
- Custom Transformers build with USM support.
- CUDA-capable USM device (`device=cuda:0` by default), like NVIDIA Jetson.
- Pre-downloaded model weights in safetensors format.

### Custom Build Instructions

Remember to install `Rust` and `cargo` beforehand with `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.

**1. Safetensors build:**

```bash
cd <your_workspace>
conda activate <env-name>

git clone https://github.com/nagic0/safetensors.git
cd safetensors
git switch dev/torch_usm
# git switch dev/torch_usm_mps # For Apple MPS devices
cd bindings/python
pip install -e .
```

**2. Transformers build:**

```bash
cd <your_workspace>
conda activate <env-name>

git clone https://github.com/nagic0/transformers.git
cd transformers
git switch dev/torch_usm
pip install -e .
```

**3. Other dependencies:**

```bash
pip install accelerate
```

## Running Correctness Tests

`correctness.py` compares the model loaded via the standard copy path and the USM-backed path and verifies equivalence. Usage:

```bash
python correctness.py --device cuda --model <model-path-or-id>
```

Notes:

- Supports both local model paths (e.g., `./Qwen3-8B`) and Hugging Face model IDs (e.g., `meta-llama/Llama-3.2-3B-Instruct`).
- The script treats exact binary equality (max diff == 0) as PASS.

### NVIDIA Jetson Example

Example on NVIDIA Jetson AGX Orin (L4T 35.6.0, Jetpack 5.1.4, CUDA 12.2):

```
> python correctness.py --device cuda --model meta-llama/Llama-3.2-3B-Instruct

...

Summary:
Total compared tensors: 254
Passed (max_diff==0): 254
Overall PASS: True
```

### AMD APU Example

**Important:**

Due to ROCm's memory allocation behavior on AMD APU platforms, HIP only recognizes VRAM allocated via the BIOS and does not detect available host memory. To fully utilize system memory for loading large models, please refer to [segurac/force-host-alloction-APU](https://github.com/segurac/force-host-alloction-APU). 

Example on AMD Strix Point (Ubuntu 24.04, Linux 6.14.0, ROCm 7.0.2):

```
> LD_PRELOAD=./libforcegttalloc.so python correctness.py --device cuda --model meta-llama/Llama-3.2-3B-Instruct

...

Summary:
Total compared tensors: 254
Passed (max_diff==0): 254
Overall PASS: True
```

### Intel iGPU Example

**Important:**

- On Intel iGPU platforms before Xe2, there's a max size limit, around 4GB, for a single GPU buffer ([this issue](https://github.com/intel/compute-runtime/issues/627)). Therefore, for larger models (2B+), consider using the `intel_resize.py` script to **resize the file sizes** of the model weights before running the tests.

Example on Intel Arrow Lake (Ubuntu 24.04, Linux 6.16.9, oneAPI 2025.2.0):

```
> python correctness.py --device xpu --model ./Llama-3.1-8B-Instruct-Resize

...

Summary:
Total compared tensors: 291
Passed (max_diff==0): 291
Overall PASS: True
```

### Apple Metal Example

Example on Apple M4 Pro (macOS 15.7):

```
> python correctness.py --device mps --model Qwen/Qwen3-8B

...

Summary:
Total compared tensors: 399
Passed (max_diff==0): 399
Overall PASS: True
```

## Running Performance Benchmark

`performance.py` automates running repeated timed trials using `performance_worker.py` to measure model loading performance. It runs a configurable number of warmup runs and measured loops per model.

Run the benchmark:

```bash
sudo ls
python performance.py --device <device> --model <model-path-or-id>
```

Notes:

- The script spawns worker subprocesses which print timing results — `performance.py` parses that to compute average latencies.
- For accurate measurements, inside the worker, we drop filesystem caches before each timed run.
- Supports both local model paths and Hugging Face model IDs.

### NVIDIA Jetson Example

Example on NVIDIA Jetson AGX Orin (L4T 35.6.0, Jetpack 5.1.4, CUDA 12.2):

```
> sudo ls
> python performance.py --device cuda --model Qwen/Qwen3-8B

Benchmarking model: Qwen/Qwen3-8B
  Running COPY [Warmup] 1/5... 19.2688 s
  Running COPY [Warmup] 2/5... 19.4247 s
  Running COPY [Timing] 3/5... 19.6518 s
  Running COPY [Timing] 4/5... 18.5372 s
  Running COPY [Timing] 5/5... 19.1089 s
  Running USM [Warmup] 1/5... 10.9389 s
  Running USM [Warmup] 2/5... 11.0145 s
  Running USM [Timing] 3/5... 11.0272 s
  Running USM [Timing] 4/5... 10.8913 s
  Running USM [Timing] 5/5... 10.8857 s

Summary for model: Qwen/Qwen3-8B
Copy average: 19.099295 s
USM average:  10.934755 s
Time Reduce: 42.75 %
```

### AMD APU Example

Example on AMD Strix Point (Ubuntu 24.04, Linux 6.14.0, ROCm 7.0.2):

```
> sudo ls
> LD_PRELOAD=./libforcegttalloc.so python performance.py --device cuda --model Qwen/Qwen3-8B

Benchmarking model: ./models/Qwen3-8B/
  Running COPY [Warmup] 1/5... 12.3647 s
  Running COPY [Warmup] 2/5... 12.4575 s
  Running COPY [Timing] 3/5... 13.3925 s
  Running COPY [Timing] 4/5... 12.4761 s
  Running COPY [Timing] 5/5... 12.3232 s
  Running USM [Warmup] 1/5... 8.2446 s
  Running USM [Warmup] 2/5... 8.3298 s
  Running USM [Timing] 3/5... 8.2258 s
  Running USM [Timing] 4/5... 8.2745 s
  Running USM [Timing] 5/5... 8.2381 s

Summary for model: ./models/Qwen3-8B/
Copy average: 12.730602 s
USM average:  8.246126 s
Time Reduce: 35.23 %
```

### Intel iGPU Example

Example on Intel Arrow Lake (Ubuntu 24.04, Linux 6.16.9, oneAPI 2025.2.0):

```
> source /opt/intel/oneapi/setvars.sh

# Resize the model and save to local path first
> python intel_resize.py --load google/gemma-3-27b-it --save ./gemma-3-27b-it-Resize 
Loading model from google/gemma-3-27b-it...
Loading checkpoint shards: 100%|███████████████| 12/12 [00:00<00:00, 124.18it/s]
Resized model saved to ./gemma-3-27b-it-Resize.

> sudo ls
> python performance.py --device xpu --model ./gemma-3-27b-it-Resize

Benchmarking model: ./gemma-3-27b-it-Resize
  Running COPY [Warmup] 1/5... 41.7727 s
  Running COPY [Warmup] 2/5... 41.5967 s
  Running COPY [Timing] 3/5... 38.8560 s
  Running COPY [Timing] 4/5... 38.8384 s
  Running COPY [Timing] 5/5... 37.5333 s
  Running USM [Warmup] 1/5... 12.3734 s
  Running USM [Warmup] 2/5... 12.8650 s
  Running USM [Timing] 3/5... 10.9618 s
  Running USM [Timing] 4/5... 11.2411 s
  Running USM [Timing] 5/5... 11.3855 s

Summary for model: ./gemma-3-27b-it-Resize
Copy average: 38.409206 s
USM average:  11.196101 s
Time Reduce: 70.85 %
```

### Apple Metal Example

Example on Apple M4 Pro (macOS 15.7):

```
> sudo ls
> python performance.py --device mps --model Qwen/Qwen3-8B

Benchmarking model: Qwen/Qwen3-8B
  Running COPY [Warmup] 1/5... 7.8868 s
  Running COPY [Warmup] 2/5... 8.2929 s
  Running COPY [Timing] 3/5... 8.3855 s
  Running COPY [Timing] 4/5... 8.6560 s
  Running COPY [Timing] 5/5... 8.6914 s
  Running USM [Warmup] 1/5... 2.0805 s
  Running USM [Warmup] 2/5... 1.9356 s
  Running USM [Timing] 3/5... 1.7765 s
  Running USM [Timing] 4/5... 1.7540 s
  Running USM [Timing] 5/5... 1.7840 s

Summary for model: Qwen/Qwen3-8B
Copy average: 8.577654 s
USM average:  1.771506 s
Time Reduce: 79.35 %
```

## Interactive Chat Demo

`chat.py` provides an interactive chat interface to test model inference with and without USM loading.

Run the chat demo:

```bash
python chat.py --device <device> --model <model-path-or-id> [--usm]
```

### NVIDIA Jetson Example

Example on NVIDIA Jetson AGX Orin (L4T 35.6.0, Jetpack 5.1.4, CUDA 12.2):

```
> python chat.py --device cuda --model Qwen/Qwen3-8B --usm

Loading model: Qwen/Qwen3-8B (USM=True)
Loading checkpoint shards: 100%|███████████████| 5/5 [00:10<00:00,  2.18s/it]
Loaded model in 11.75 seconds.

User: Who are you?
Assistant:  I am Qwen, a large-scale language model developed by Alibaba Cloud. I was trained on a vast amount of text data and can assist with a wide range of tasks, such as answering questions, creating content, and providing information.

User: exit
Exiting...
```

### AMD APU Example

Example on AMD Strix Point (Ubuntu 24.04, Linux 6.14.0, ROCm 7.0.2):

```
> TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 LD_PRELOAD=./libforcegttalloc.so python chat.py --device cuda --model Qwen/Qwen3-8B --usm
Loading model: Qwen/Qwen3-8B (USM=True)

User: Who are you
Assistant:  I am Qwen, a large language model developed by Alibaba Cloud. I can assist with various tasks such as answering questions, creating content, and providing programming help.

User: exit
Exiting...
```

## Chat Performance Measurement

`chat_performance.py` measures the loading and inference latency of the chat model with and without USM.

Run the chat performance measurement:

```bash
> sudo ls
> python chat_performance.py --device <device> --model <model-path-or-id>
```

### NVIDIA Jetson Example

Example on NVIDIA Jetson AGX Orin (L4T 35.6.0, Jetpack 5.1.4, CUDA 12.2):

```
> sudo ls
> python chat_performance.py --device cuda --model Llama/Llama-3.2-3B-Instruct

...

============================================================
SUMMARY COMPARISON
============================================================
Metric                    No USM          With USM       
------------------------------------------------------------
Load time (s)             8.4974          4.5436         
Prefill time (s)          0.1088          0.1065         
Decode time (s)           2.6240          2.5765         
Total time (s)            2.7328          2.6831         
Generated tokens          33              33             
Decode throughput (tok/s) 12.58           12.81          
============================================================
```

### AMD APU Example

Example on AMD Strix Point (Ubuntu 24.04, Linux 6.14.0, ROCm 7.0.2):

```
> sudo ls
> LD_PRELOAD=./libforcegttalloc.so python chat_performance.py --device cuda --model Qwen/Qwen3-8B

============================================================
SUMMARY COMPARISON
============================================================
Metric                    No USM          With USM       
------------------------------------------------------------
Load time (s)             16.8431         8.1556         
Prefill time (s)          0.2908          0.2667         
Decode time (s)           6.7902          7.9225         
Total time (s)            7.0810          8.1892         
Generated tokens          32              32             
Decode throughput (tok/s) 4.71            4.04           
============================================================
```

### Intel iGPU Example

Example on Intel Arrow Lake (Ubuntu 24.04, Linux 6.16.9, oneAPI 2025.2.0):

```
> sudo ls
> python chat_performance.py --device xpu --model ./Meta-Llama-3.1-8B-Instruct-Resize

...

============================================================
SUMMARY COMPARISON
============================================================
Metric                    No USM          With USM       
------------------------------------------------------------
Load time (s)             12.4229         3.2007         
Prefill time (s)          0.2706          0.2816         
Decode time (s)           6.0887          6.4212         
Total time (s)            6.3594          6.7028         
Generated tokens          24              24             
Decode throughput (tok/s) 3.94            3.74           
============================================================
```

### Apple Metal Example

**Important:**

On Apple MPS devices, USM allocation does not support huge pages, which may lead to suboptimal performance compared to the default GPU allocator. 

Example on Apple M4 Pro (macOS 15.7):

```
> sudo ls
> python chat_performance.py --device mps --model Qwen/Qwen3-8B

.../transformers/src/transformers/modeling_utils.py:749: UserWarning: USM: macOS does not support allocating huge pages. Performance may be suboptimal compared to the default GPU allocator. (Triggered internally at .../pytorch/aten/src/ATen/UsmAllocator.cpp:85.)
  file_pointer = safe_open(shard_file, framework="pt", usm_device=_usm_device)

============================================================
SUMMARY COMPARISON
============================================================
Metric                    No USM          With USM       
------------------------------------------------------------
Load time (s)             8.4910          5.3629         
Prefill time (s)          0.1817          0.6470         
Decode time (s)           2.3194          3.0272         
Total time (s)            2.5011          3.6743         
Generated tokens          32              32             
Decode throughput (tok/s) 13.80           10.57          
============================================================
```