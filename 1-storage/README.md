# USM Loading Test on Storage

This directory contains small tests and benchmarks to evaluate loading binary weight files into PyTorch using two methods:

- Standard copy-based file->CPU->GPU path.
- USM-backed storage (Unified Shared Memory) path that aims to avoid an explicit host copy.

The tests are intended for systems with custom PyTorch builds that expose `UntypedStorage.from_file(..., usm=True)` and `usm_share_()` APIs.

## Repository layout

- `correctness.py` — Verifies binary equality between the two loading methods.
- `performance.py` — Runs an automated micro-benchmark across multiple file sizes.
- `performance_worker.py` — Worker script invoked by `performance.py` for a single timed run.

## Prerequisites

- Python 3.8+ with `numpy` and `torch` installed.
- A PyTorch build that supports USM storage APIs (custom builds).
- CUDA-capable USM device (`device=cuda:0` by default), like NVIDIA Jetson.

Install runtime dependencies:

```bash
pip install numpy
```

## Running Correctness Tests

`correctness.py` compares the data loaded via the standard copy path and the USM-backed path and prints the maximum absolute difference. Usage:

```bash
python correctness.py
```

Example results (measured on NVIDIA Jetson AGX Orin, L4T 35.6.0, Jetpack 5.1.4, CUDA 12.2):

```
Size (MB)    | Status   | Max Diff  
----------------------------------------
0.001        | ✅ PASS   | 0.0000e+00
10           | ✅ PASS   | 0.0000e+00
100          | ✅ PASS   | 0.0000e+00
500          | ✅ PASS   | 0.0000e+00
1000         | ✅ PASS   | 0.0000e+00
```

Notes:

- `correctness.py` creates temporary binary files and removes them after the check.
- The script treats exact binary equality (max diff == 0) as PASS.

## Running Performance Benchmark

`performance.py` automates generating test files and running repeated timed trials using `performance_worker.py`. It runs a configurable number of warmup runs and measured loops per size.

Run the full benchmark:

```bash
python performance.py
```

Important:

- The script spawns worker subprocesses which print a `RESULT: <seconds>` line — `performance.py` parses that to compute average latencies.
- For accurate measurements, inside the worker, we drop filesystem caches before each timed run. This may require root privileges depending on your system configuration. To avoid password prompts during the benchmark, you can pre-authenticate with:

```bash
sudo ls  # To get root privileges without password prompt later
python performance.py
```

Example results (measured on NVIDIA Jetson AGX Orin, L4T 35.6.0, Jetpack 5.1.4, CUDA 12.2):

```
============================================================
                   FINAL RESULTS SUMMARY                    
============================================================
Size (MB)  | Copy Time (s)   | USM Time (s)    | Time Reduce (%)
------------------------------------------------------------
1          | 0.006555        | 0.003331        | 49.18     
100        | 0.117807        | 0.107668        | 8.61      
1000       | 1.031310        | 0.826118        | 19.90     
5000       | 5.208359        | 3.412800        | 34.47
```
