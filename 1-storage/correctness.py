import torch
import os
from pathlib import Path

ROOT = Path(__file__).parent
device = torch.device("cuda")

def get_usm(filename, num_elements):
    file_size = os.path.getsize(filename)
    # shared=False, usm=True implies internal USM allocation handling
    storage = torch.UntypedStorage.from_file(filename, shared=False, nbytes=file_size, usm=True)
    
    storage_gpu = storage.usm_share_(device=device)
    tensor_gpu = torch.empty(num_elements, dtype=torch.float32, device=device)
    tensor_gpu.set_(storage_gpu)

    loaded_tensor = torch.empty(num_elements, dtype=torch.float32)
    loaded_tensor.set_(storage)
    
    return loaded_tensor

def get_copy(filename, num_elements):
    file_size = os.path.getsize(filename)
    storage = torch.UntypedStorage.from_file(filename, shared=False, nbytes=file_size)
    
    storage_gpu = storage.to(device=device)
    tensor_gpu = torch.empty(num_elements, dtype=torch.float32, device=device)
    tensor_gpu.set_(storage_gpu)

    loaded_tensor = torch.empty(num_elements, dtype=torch.float32)
    loaded_tensor.set_(storage)
    
    return loaded_tensor

def test_correctness(size_mb):
    num_elements = int((size_mb * 1024 * 1024) // 4)
    filename = ROOT / f"weights_{size_mb}mb.bin"
    
    # Create temp file
    tensor_cpu = torch.randn(num_elements, dtype=torch.float32)
    tensor_cpu.numpy().tofile(filename)
    del tensor_cpu

    import gc
    gc.collect()

    try:
        copy_tensor = get_copy(filename.as_posix(), num_elements)
        usm_tensor = get_usm(filename.as_posix(), num_elements)

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
    sizes = [0.001, 10, 100, 500, 1000] 

    # Print Header
    print(f"{'Size (MB)':<12} | {'Status':<8} | {'Max Diff':<10}")
    print("-" * 40)

    for size in sizes:
        test_correctness(size)