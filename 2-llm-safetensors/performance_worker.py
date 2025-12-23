import os
from pathlib import Path
import time
import argparse
import sys
import torch
from contextlib import nullcontext
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import set_usm_device

ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(ROOT))

def device_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    else:
        pass

def device_empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    else:
        pass

def flush(device):
    device_empty_cache(device)
    os.system("sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null")

def load_model(path: str, use_usm: bool, device_map="cuda"):
    flush(torch.device(device_map))

    t0 = time.time()
    loader_context = set_usm_device(device_map) if use_usm else nullcontext()
    with loader_context:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype="auto",
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    device_sync(torch.device(device_map))
    t1 = time.time()

    return model, t1 - t0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to pretrained model folder")
    parser.add_argument("--method", choices=["copy", "usm"], required=True)
    parser.add_argument("--device", "-d", type=str, required=True, help="Device to use (e.g., cuda, xpu)")
    args = parser.parse_args()

    try:
        use_usm = args.method == "usm"
        torch.ones(1).to(device=args.device)
        device_sync(torch.device(args.device))

        model, duration = load_model(args.model, use_usm=use_usm, device_map=args.device)
        print(f"RESULT: {duration:.6f}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
