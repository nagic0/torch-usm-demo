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

- Python 3.8+ with `numpy`, `torch`, and `transformers` installed.
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

Install runtime dependencies:

```bash
pip install numpy
```

## Running Correctness Tests

`correctness.py` compares the model loaded via the standard copy path and the USM-backed path and verifies equivalence. Usage:

```bash
python correctness.py --model <model-path-or-id>
```

Example output:

```
> python ./2-llm-safetensors/correctness.py --model meta-llama/Llama-3.2-3B-Instruct

...

Summary:
Total compared tensors: 254
Passed (max_diff==0): 254
Overall PASS: True
```

Notes:

- Supports both local model paths (e.g., `./Qwen3-8B`) and Hugging Face model IDs (e.g., `meta-llama/Llama-3.2-3B-Instruct`).
- The script treats exact binary equality (max diff == 0) as PASS.

## Running Performance Benchmark

`performance.py` automates running repeated timed trials using `performance_worker.py` to measure model loading performance. It runs a configurable number of warmup runs and measured loops per model.

Run the benchmark:

```bash
python performance.py --model <model-path-or-id>
```

Example output:

```
> python ./2-llm-safetensors/performance.py --model Qwen/Qwen3-8B

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

Summary for model: ./Qwen3-8B
Copy average: 19.099295 s
USM average:  10.934755 s
Time Reduce: 42.75 %
```

Important:

- The script spawns worker subprocesses which print timing results — `performance.py` parses that to compute average latencies.
- For accurate measurements, inside the worker, we drop filesystem caches before each timed run.
- Supports both local model paths and Hugging Face model IDs.

## Interactive Chat Demo

`chat.py` provides an interactive chat interface to test model inference with and without USM loading.

Run without USM:

```bash
python chat.py --model Qwen/Qwen3-8B
```

Run with USM:

```bash
python chat.py --model Qwen/Qwen3-8B --usm
```

Example output (without USM):

```
> python ./2-llm-safetensors/chat.py --model Qwen/Qwen3-8B

Loading model: Qwen/Qwen3-8B (USM=False)
Loading checkpoint shards: 100%|███████████████| 5/5 [00:15<00:00,  3.02s/it]
Loaded model in 19.82 seconds.

User: Who are you?
Assistant:  I am Qwen, a large-scale language model developed by Alibaba Cloud. I was trained on a vast amount of text data and can understand and generate human-like text. My capabilities include answering questions, creating content, and engaging in conversations.

User: exit
Exiting...
```

Example output (with USM):

```
> python ./2-llm-safetensors/chat.py --model Qwen/Qwen3-8B --usm

Loading model: Qwen/Qwen3-8B (USM=True)
Loading checkpoint shards: 100%|███████████████| 5/5 [00:10<00:00,  2.18s/it]
Loaded model in 11.75 seconds.

User: Who are you?
Assistant:  I am Qwen, a large-scale language model developed by Alibaba Cloud. I was trained on a vast amount of text data and can assist with a wide range of tasks, such as answering questions, creating content, and providing information.

User: exit
Exiting...
```
