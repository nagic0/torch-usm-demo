import torch
import os
import time
import argparse
import sys


def flush():
    torch.cuda.empty_cache()
    os.system("sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null")

def run_usm(filename, device):
    file_size = os.path.getsize(filename)
    t0 = time.time()
    storage = torch.UntypedStorage.from_file(filename, shared=False, nbytes=file_size, usm=True)
    load_end = time.time()
    storage_gpu = storage.usm_share_(device=device)
    torch.cuda.synchronize()
    t_end = time.time()
    
    return t_end - t0

def run_copy(filename, device):
    file_size = os.path.getsize(filename)
    t0 = time.time()
    storage = torch.UntypedStorage.from_file(filename, shared=False, nbytes=file_size)
    load_end = time.time()
    storage_gpu = storage.to(device=device)
    torch.cuda.synchronize()
    t_end = time.time()
    
    return t_end - t0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to binary file")
    parser.add_argument("--method", type=str, choices=["copy", "usm"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    # warmup context
    torch.ones(1).to(device)
    torch.cuda.synchronize()

    flush()

    try:
        if args.method == "copy":
            duration = run_copy(args.file, device)
        elif args.method == "usm":
            duration = run_usm(args.file, device)
        print(f"RESULT: {duration:.6f}")
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)