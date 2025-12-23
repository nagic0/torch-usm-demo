import os
import torch


def device_empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    else:
        pass


def flush_page_cache():
    # Mac 
    if os.uname().sysname == "Darwin":
        os.system("sudo sync; sudo purge")
    # Linux
    else:
        os.system("sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null")


def flush(device):
    device_empty_cache(device)
    flush_page_cache()

def device_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    else:
        pass

