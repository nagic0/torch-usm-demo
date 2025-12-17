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

## Prerequisites

- Python 3.8+ with `numpy` and `accelerate` installed.
- Custom PyTorch build with USM storage support (see [main README](../README.md)).
- Custom Safetensors build with USM storage support.
- Custom Transformers build with USM support.
- CUDA-capable USM device (`device=cuda:0` by default), like NVIDIA Jetson.
- Pre-downloaded model weights in safetensors format.

### Custom Build Instructions

Remember to install `Rust` and `cargo` beforehand with `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.

**Safetensors build:**

```bash
cd <your_workspace>
conda activate <env-name>

git clone https://github.com/nagic0/safetensors.git
cd safetensors
git switch dev/torch_usm
cd bindings/python
pip install -e .
```

**Transformers build:**

```bash
cd <your_workspace>
conda activate <env-name>

git clone https://github.com/nagic0/transformers.git
cd transformers
git switch dev/torch_usm
pip install -e .
```

Install prequisites:

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

### Intel iGPU Example

**Important:**

- On Intel iGPU platforms before Xe2, there's a max size limit, around 4GB, for a single GPU buffer (See [this issue](https://github.com/intel/compute-runtime/issues/627)). Therefore, for larger models (2B+), consider using the `intel_resize.py` script to **resize the file sizes** of the model weights before running the tests.

Example on Intel Arrow Lake (Ubuntu 24.04, Linux 6.16.9, oneAPI 2025.2.0):

```
> python correctness.py --device xpu --model ./Llama-3.1-8B-Instruct-Resize

...

Summary:
Total compared tensors: 291
Passed (max_diff==0): 291
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

## Interactive Chat Demo

`chat.py` provides an interactive chat interface to test model inference with and without USM loading.

Run the chat demo:

```bash
python chat.py --device <device> --model <model-path-or-id> [--usm]
```

### NVIDIA Jetson Example

Example on NVIDIA Jetson AGX Orin (L4T 35.6.0, Jetpack 5.1.4, CUDA 12.2):

```
> python chat.py --device cuda --model Qwen/Qwen3-8B

Loading model: Qwen/Qwen3-8B (USM=False)
Loading checkpoint shards: 100%|███████████████| 5/5 [00:15<00:00,  3.02s/it]
Loaded model in 19.82 seconds.

User: Who are you?
Assistant:  I am Qwen, a large-scale language model developed by Alibaba Cloud. I was trained on a vast amount of text data and can understand and generate human-like text. My capabilities include answering questions, creating content, and engaging in conversations.

User: exit
Exiting...

> python chat.py --device cuda --model Qwen/Qwen3-8B --usm

Loading model: Qwen/Qwen3-8B (USM=True)
Loading checkpoint shards: 100%|███████████████| 5/5 [00:10<00:00,  2.18s/it]
Loaded model in 11.75 seconds.

User: Who are you?
Assistant:  I am Qwen, a large-scale language model developed by Alibaba Cloud. I was trained on a vast amount of text data and can assist with a wide range of tasks, such as answering questions, creating content, and providing information.

User: exit
Exiting...
```

### Intel iGPU Example

Due to the [AOT compilation issue or incorrect use by me](https://github.com/intel/torch-xpu-ops/issues/2587) on Intel iGPU platforms, the chat demo may experience significant delays during the first inference call. Subsequent calls should be faster.

*To be continued...*