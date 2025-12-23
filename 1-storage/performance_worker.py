from pathlib import Path
import torch
import os
import time
import argparse
import sys

ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT.parent))

from common import flush, device_sync

def run_usm(filename, device):
    file_size = os.path.getsize(filename)
    t0 = time.time()
    storage = torch.UntypedStorage.from_file(filename, shared=False, nbytes=file_size, usm=True)
    load_end = time.time()
    storage_gpu = storage.usm_share_(device=device)
    device_sync(device)
    t_end = time.time()
    
    return t_end - t0

def run_copy(filename, device):
    file_size = os.path.getsize(filename)
    t0 = time.time()
    storage = torch.UntypedStorage.from_file(filename, shared=False, nbytes=file_size)
    load_end = time.time()
    storage_gpu = storage.to(device=device)
    device_sync(device)
    t_end = time.time()
    
    return t_end - t0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to binary file")
    parser.add_argument("--method", type=str, choices=["copy", "usm"], required=True)
    parser.add_argument("--device", "-d", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)

    # warmup context
    torch.ones(1).to(device)
    device_sync(device)

    flush(device)

    try:
        if args.method == "copy":
            duration = run_copy(args.file, device)
        elif args.method == "usm":
            duration = run_usm(args.file, device)
        print(f"RESULT: {duration:.6f}")
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)