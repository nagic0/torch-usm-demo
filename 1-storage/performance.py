import os
import torch
import subprocess
import numpy as np
from pathlib import Path
import re
import sys
import argparse

ROOT = Path(__file__).parent.absolute()
WORKER_SCRIPT = ROOT / "performance_worker.py"

def create_dummy_file(filename, size_mb):
    num_elements = int((size_mb * 1024 * 1024) // 4)
    print(f"Generating {size_mb} MB file at {filename}...", end=" ", flush=True)
    arr = np.random.randn(num_elements).astype(np.float32)
    arr.tofile(filename)
    del arr
    print("Done.")

def run_single_test(filename, method, loop, warmup, device):
    latencies = []

    total_runs = warmup + loop
    
    for i in range(total_runs):
        is_warmup = i < warmup
        prefix = "[Warmup]" if is_warmup else "[Timing]"
        print(f"  Running {method.upper()} {prefix} {i+1}/{total_runs}...", end="", flush=True)

        cmd = [
            sys.executable, str(WORKER_SCRIPT),
            "--file", str(filename),
            "--method", method,
            "--device", device
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            
            match = re.search(r"RESULT:\s*([0-9\.]+)", output)
            if match:
                duration = float(match.group(1))
                print(f" {duration:.4f} s")
                
                if not is_warmup:
                    latencies.append(duration)
            else:
                print(f" Failed to parse output. Output: {output}")

        except subprocess.CalledProcessError as e:
            print(f"\n  Error running worker: {e.stderr}")
            return None

    if not latencies:
        return 0.0

    return sum(latencies) / len(latencies)

def main():
    parser = argparse.ArgumentParser(description='Performance test for USM vs Copy')
    parser.add_argument('--device', "-d", type=str, required=True,
                        help='Device to use (e.g., cuda, xpu)')
    args = parser.parse_args()

    # check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA device is not available. Exiting.")
        exit(1)
    elif args.device == 'xpu' and not torch.xpu.is_available():
        print("XPU device is not available. Exiting.")
        exit(1)
    
    print(f"Using device: {args.device}")
    print()
    
    sizes = [1, 100, 1000, 4000]
    warmup = 2
    loop = 3

    results = {}

    if not WORKER_SCRIPT.exists():
        print(f"Error: Could not find {WORKER_SCRIPT}")
        return

    for size in sizes:
        print(f"\n{'='*40}")
        print(f"Starting Benchmark for Size: {size} MB")
        print(f"{'='*40}")
        
        filename = ROOT / f"weights_{size}mb.bin"
        
        try:
            create_dummy_file(filename, size)
            copy_avg = run_single_test(filename, "copy", loop, warmup, args.device)
            usm_avg = run_single_test(filename, "usm", loop, warmup, args.device)

            results[size] = (copy_avg, usm_avg)

        finally:
            if filename.exists():
                os.remove(filename)
                print(f"Cleaned up {filename}")

    print(f"\n\n{'='*60}")
    print(f"{'FINAL RESULTS SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"{'Size (MB)':<10} | {'Copy Time (s)':<15} | {'USM Time (s)':<15} | {'Time Reduce (%)':<10}")
    print("-" * 60)
    
    for size in sizes:
        copy_t, usm_t = results[size]
        speedup = 0.0
        if copy_t > 0:
            speedup = (copy_t - usm_t) / copy_t * 100
        
        print(f"{size:<10} | {copy_t:<15.6f} | {usm_t:<15.6f} | {speedup:<10.2f}")

if __name__ == "__main__":
    main()