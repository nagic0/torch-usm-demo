import os
import time
import argparse
import sys
import torch
from contextlib import nullcontext
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import set_usm_device


def flush():
    torch.cuda.empty_cache()
    os.system("sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null")

def load_model(path: str, use_usm: bool, device_map="cuda"):
    flush()

    t0 = time.time()
    loader_context = set_usm_device("cuda") if use_usm else nullcontext()
    with loader_context:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype="auto",
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    # ensure CUDA work is finished before measuring
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    return model, t1 - t0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to pretrained model folder")
    parser.add_argument("--method", choices=["copy", "usm"], required=True)
    parser.add_argument("--device_map", type=str, default="cuda")
    args = parser.parse_args()

    try:
        use_usm = args.method == "usm"
        # warmup small tensor to init CUDA
        if torch.cuda.is_available():
            torch.ones(1).to('cuda')
            torch.cuda.synchronize()

        model, duration = load_model(args.model, use_usm=use_usm, device_map=args.device_map)
        print(f"RESULT: {duration:.6f}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
