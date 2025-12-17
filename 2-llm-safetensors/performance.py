import os
import sys
import re
import subprocess
import argparse
from pathlib import Path


ROOT = Path(__file__).parent.absolute()
WORKER_SCRIPT = ROOT / "performance_worker.py"


def run_single_test(model_path, method, loop, warmup, device_map):
    latencies = []
    total_runs = warmup + loop

    for i in range(total_runs):
        is_warmup = i < warmup
        prefix = "[Warmup]" if is_warmup else "[Timing]"
        print(f"  Running {method.upper()} {prefix} {i+1}/{total_runs}...", end="", flush=True)

        cmd = [
            sys.executable, str(WORKER_SCRIPT),
            "--model", str(model_path),
            "--method", method,
            "--device", device_map,
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
    parser = argparse.ArgumentParser(description="Benchmark model load time for copy vs USM")
    parser.add_argument("--model", "-m", required=True, help="Path to pretrained model folder")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--loop", type=int, default=3)
    parser.add_argument("--device", "-d", type=str, required=True, help="Device to use (e.g., cuda, xpu)")
    args = parser.parse_args()

    if not WORKER_SCRIPT.exists():
        print(f"Error: Could not find {WORKER_SCRIPT}")
        return

    print(f"Benchmarking model: {args.model}")

    copy_avg = run_single_test(args.model, "copy", args.loop, args.warmup, args.device)
    usm_avg = run_single_test(args.model, "usm", args.loop, args.warmup, args.device)

    print(f"\nSummary for model: {args.model}")
    print(f"Copy average: {copy_avg:.6f} s")
    print(f"USM average:  {usm_avg:.6f} s")
    if copy_avg > 0:
        speedup = (copy_avg - usm_avg) / copy_avg * 100
    else:
        speedup = 0.0
    print(f"Time Reduce: {speedup:.2f} %")


if __name__ == "__main__":
    main()
