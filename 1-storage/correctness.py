import torch
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).parent

def get_usm(filename, num_elements, device):
    file_size = os.path.getsize(filename)
    # shared=False, usm=True implies internal USM allocation handling
    storage = torch.UntypedStorage.from_file(filename, shared=False, nbytes=file_size, usm=True)
    
    storage_gpu = storage.usm_share_(device=device)
    tensor_gpu = torch.empty(num_elements, dtype=torch.float32, device=device)
    tensor_gpu.set_(storage_gpu)

    # loaded_tensor = torch.empty(num_elements, dtype=torch.float32)
    # loaded_tensor.set_(storage)
    loaded_tensor = tensor_gpu
    
    return loaded_tensor

def get_copy(filename, num_elements, device):
    file_size = os.path.getsize(filename)
    storage = torch.UntypedStorage.from_file(filename, shared=False, nbytes=file_size)
    
    storage_gpu = storage.to(device=device)
    tensor_gpu = torch.empty(num_elements, dtype=torch.float32, device=device)
    tensor_gpu.set_(storage_gpu)

    # loaded_tensor = torch.empty(num_elements, dtype=torch.float32)
    # loaded_tensor.set_(storage)
    loaded_tensor = tensor_gpu

    return loaded_tensor

def test_correctness(size_mb, device):
    num_elements = int((size_mb * 1024 * 1024) // 4)
    filename = ROOT / f"weights_{size_mb}mb.bin"
    
    # Create temp file
    tensor_cpu = torch.randn(num_elements, dtype=torch.float32)
    tensor_cpu.numpy().tofile(filename)
    del tensor_cpu

    import gc
    gc.collect()

    try:
        copy_tensor = get_copy(filename.as_posix(), num_elements, device)
        usm_tensor = get_usm(filename.as_posix(), num_elements, device)

        max_diff = torch.abs(usm_tensor - copy_tensor).max().item()
        is_passed = max_diff == 0  # Strict equality for binary consistency check
        
        status_icon = "\u2705" if is_passed else "\u274C"
        status_text = "PASS" if is_passed else "FAIL"
        
        # Formatted output row
        print(f"{size_mb:<12} | {status_icon} {status_text:<6} | {max_diff:.4e}")

    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test USM correctness')
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
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print()
    
    sizes = [0.001, 10, 100, 500, 1000, 4000] 

    # Print Header
    print(f"{'Size (MB)':<12} | {'Status':<8} | {'Max Diff':<10}")
    print("-" * 40)

    for size in sizes:
        test_correctness(size, device)