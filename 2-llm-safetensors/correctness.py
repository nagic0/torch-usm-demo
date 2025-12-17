import os
import argparse
import gc
import torch
from contextlib import nullcontext
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import set_usm_device


def load_model(path: str, use_usm: bool, device_map):
    loader_context = set_usm_device(device_map) if use_usm else nullcontext()
    
    with loader_context:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype="auto",
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    return model


def compare_models(model_a, model_b):
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    common = sorted(set(params_a.keys()) & set(params_b.keys()))

    results = []

    for name in common:
        a = params_a[name].detach().cpu()
        b = params_b[name].detach().cpu()

        if a.shape != b.shape:
            results.append((name, None, f"SHAPE_MISMATCH: {a.shape} vs {b.shape}"))
            print(f"{name}: SHAPE_MISMATCH {a.shape} vs {b.shape}")
            continue

        # compute max absolute difference in float32 to avoid dtype saturation
        diff = torch.max(torch.abs(a.to(torch.float32) - b.to(torch.float32))).item()
        status = "PASS" if diff == 0.0 else "FAIL"
        results.append((name, diff, status))
        print(f"{name} | shape={tuple(a.shape)} | max_diff={diff:.6e} | {status}")

    # report params only in one model
    only_a = sorted(set(params_a.keys()) - set(params_b.keys()))
    only_b = sorted(set(params_b.keys()) - set(params_a.keys()))
    for name in only_a:
        print(f"{name}: ONLY_IN_A")
    for name in only_b:
        print(f"{name}: ONLY_IN_B")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare model parameters between copy and USM/shared loads")
    parser.add_argument("--model", "-m", required=True, help="Path to pretrained model folder")
    parser.add_argument("--device", "-d", type=str, required=True, help="Device to use (e.g., cuda, xpu)")

    args = parser.parse_args()

    print(f"Loading copy-based model from: {args.model}")
    # Use the model's default dtype/configuration by not passing `dtype`
    model_copy = load_model(args.model, use_usm=False, device_map=args.device)

    # free any temporary CPU memory before the second load
    gc.collect()

    print(f"Loading USM/shared model from: {args.model} (USM_DEVICE={args.device})")
    model_shared = load_model(args.model, use_usm=True, device_map=args.device)

    print("\nComparing parameters:\n")
    results = compare_models(model_copy, model_shared)

    total = len(results)
    passed = sum(1 for _, diff, status in results if status == "PASS")

    print("\nSummary:")
    print(f"Total compared tensors: {total}")
    print(f"Passed (max_diff==0): {passed}")

    overall = (passed == total) and total > 0
    print(f"Overall PASS: {overall}")

    # exit code non-zero on failure
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()