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
- CUDA-capable USM device (`device=cuda` by default), like NVIDIA Jetson.
- CUDA-capable USM device (`device=cuda:0` by default), like NVIDIA Jetson.

## Running Correctness Tests

`correctness.py` compares the data loaded via the standard copy path and the USM-backed path and prints the maximum absolute difference. Usage:

```bash
python correctness.py --device <device>
```

Notes:

- `correctness.py` creates temporary binary files and removes them after the check.
- The script treats exact binary equality (max diff == 0) as PASS.

### NVIDIA Jetson Example

Example on NVIDIA Jetson AGX Orin (L4T 35.6.0, Jetpack 5.1.4, CUDA 12.2):

```
> python correctness.py --device cuda

Using device: cuda

Size (MB)    | Status   | Max Diff  
----------------------------------------
0.001        | ✅ PASS   | 0.0000e+00
10           | ✅ PASS   | 0.0000e+00
100          | ✅ PASS   | 0.0000e+00
500          | ✅ PASS   | 0.0000e+00
1000         | ✅ PASS   | 0.0000e+00
```

### AMD APU Example

Example on AMD Strix Point (Ubuntu 24.04, Linux 6.14.0, ROCm 7.0.2):

```
> python correctness.py --device cuda

Using device: cuda

Size (MB)    | Status   | Max Diff  
----------------------------------------
0.001        | ✅ PASS   | 0.0000e+00
10           | ✅ PASS   | 0.0000e+00
100          | ✅ PASS   | 0.0000e+00
500          | ✅ PASS   | 0.0000e+00
1000         | ✅ PASS   | 0.0000e+00
```

### Intel iGPU Example

Example on Intel Arrow Lake (Ubuntu 24.04, Linux 6.16.9, oneAPI 2025.2.0):

```
> python correctness.py --device xpu

Using device: xpu

Size (MB)    | Status   | Max Diff  
----------------------------------------
0.001        | ✅ PASS   | 0.0000e+00
10           | ✅ PASS   | 0.0000e+00
100          | ✅ PASS   | 0.0000e+00
500          | ✅ PASS   | 0.0000e+00
1000         | ✅ PASS   | 0.0000e+00
4000         | ✅ PASS   | 0.0000e+00
```

### Apple Metal Example

Example on Apple M4 Pro (macOS 15.7):

```
> python correctness.py --device mps

Using device: mps

Size (MB)    | Status   | Max Diff  
----------------------------------------
0.001        | ✅ PASS   | 0.0000e+00
10           | ✅ PASS   | 0.0000e+00
100          | ✅ PASS   | 0.0000e+00
500          | ✅ PASS   | 0.0000e+00
1000         | ✅ PASS   | 0.0000e+00
4000         | ✅ PASS   | 0.0000e+00
```

## Running Performance Benchmark

`performance.py` automates generating test files and running repeated timed trials using `performance_worker.py`. It runs a configurable number of warmup runs and measured loops per size.

Notes:

- The script spawns worker subprocesses which print a `RESULT: <seconds>` line — `performance.py` parses that to compute average latencies.
- For accurate measurements, inside the worker, we drop filesystem caches before each timed run. This may require root privileges depending on your system configuration. To avoid password prompts during the benchmark, you can pre-authenticate with:

```bash
sudo ls  # To get root privileges without password prompt later
python performance.py --device <device>
```

### NVIDIA Jetson Example

Example results (measured on NVIDIA Jetson AGX Orin, L4T 35.6.0, Jetpack 5.1.4, CUDA 12.2):

```
> sudo ls
> python performance.py --device cuda

Using device: cuda

...

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

### AMD APU Example

Example on AMD Strix Point (Ubuntu 24.04, Linux 6.14.0, ROCm 7.0.2):

```
> sudo ls
> python performance.py --device cuda

...

============================================================
                   FINAL RESULTS SUMMARY                    
============================================================
Size (MB)  | Copy Time (s)   | USM Time (s)    | Time Reduce (%)
------------------------------------------------------------
1          | 0.009638        | 0.001128        | 88.30     
100        | 0.086465        | 0.050183        | 41.96     
1000       | 0.793575        | 0.498848        | 37.14     
4000       | 3.222825        | 2.005964        | 37.76
```

### Intel iGPU Example

Example results (measured on Intel Arrow Lake, Ubuntu 24.04, Linux 6.16.9, oneAPI 2025.2.0):

```
> sudo ls
> python performance.py --device xpu

Using device: xpu

...

============================================================
                   FINAL RESULTS SUMMARY                    
============================================================
Size (MB)  | Copy Time (s)   | USM Time (s)    | Time Reduce (%)
------------------------------------------------------------
1          | 0.005874        | 0.001820        | 69.02     
100        | 0.071528        | 0.040415        | 43.50     
1000       | 0.592847        | 0.220699        | 62.77     
4000       | 2.290023        | 0.864745        | 62.24
```

### Apple Metal Example

Example on Apple M4 Pro (macOS 15.7):

```
> sudo ls
> python performance.py --device mps

Using device: mps

...

============================================================
                   FINAL RESULTS SUMMARY                    
============================================================
Size (MB)  | Copy Time (s)   | USM Time (s)    | Time Reduce (%)
------------------------------------------------------------
1          | 0.034657        | 0.022242        | 35.82     
100        | 0.103381        | 0.038576        | 62.69     
1000       | 0.495429        | 0.222136        | 55.16     
4000       | 1.857312        | 0.804180        | 56.70 
```