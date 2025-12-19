import os
import time
import threading
import argparse
import sys
from contextlib import nullcontext
import numpy as np

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.modeling_utils import set_usm_device


def flush(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    else:
        pass
    os.system("sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null")

def device_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    else:
        pass

def load_model(path: str, use_usm: bool, device_map="cuda"):
    flush(torch.device(device_map))

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    loader_context = set_usm_device(device_map) if use_usm else nullcontext()
    with loader_context:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype="auto",
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    # ensure pad_token_id is set on both tokenizer and model to avoid open-end generation warnings
    eos_id = None
    eos_id = getattr(tokenizer, "eos_token_id", None) or getattr(model.config, "eos_token_id", None)
    if getattr(tokenizer, "pad_token_id", None) is None and eos_id is not None:
        try:
            # prefer setting pad_token string if available to keep tokenizer state consistent
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token_id = eos_id
        except Exception:
            tokenizer.pad_token_id = eos_id
    if getattr(model.config, "pad_token_id", None) is None and eos_id is not None:
        model.config.pad_token_id = eos_id
    device_sync(torch.device(device_map))
    t1 = time.time()

    return tokenizer, model, t1 - t0


def stream_generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, verbose=True):
    # conversational streaming: prompt should already include "User:" and trailing "Assistant:" marker
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
    
    prompt_len = input_ids.shape[1]

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # match with and without leading newline to catch different model outputs
    stop_markers = ["\nUser:", "User:", "\nAssistant:", "Assistant:"]

    class StopOnMarker(StoppingCriteria):
        def __init__(self, tokenizer, markers, prompt_len: int):
            self.tokenizer = tokenizer
            self.markers = markers
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores, **kwargs):
            # only decode the newly generated suffix (avoid matching markers present in prompt)
            try:
                # input_ids shape: (1, seq_len)
                seq = input_ids[0]
                gen_ids = seq[self.prompt_len :]
                if gen_ids.numel() == 0:
                    return False
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            except Exception:
                return False
            for m in self.markers:
                if m in text:
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnMarker(tokenizer, stop_markers, prompt_len)])

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "streamer": streamer,
        "stopping_criteria": stopping_criteria,
    }

    device_sync(device)
    prefill_start = time.time()
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # stream with detection of conversation delimiters to avoid model emitting the next "User:" turn
    buffer = ""
    printed_len = 0
    collected = []
    stop_markers = ["\nUser:", "User:", "\nAssistant:", "Assistant:"]
    first_token_time = None
    chunk_times = []

    try:
        for chunk in streamer:
            if first_token_time is None:
                device_sync(device)
                first_token_time = time.time()
            
            chunk_times.append(time.time())
            buffer += chunk

            # check for any stop marker in the buffer
            idx = -1
            for m in stop_markers:
                i = buffer.find(m)
                if i != -1:
                    idx = i
                    break

            if idx != -1:
                # print up to the marker and stop
                to_print = buffer[printed_len:idx]
                if to_print:
                    if verbose:
                        print(to_print, end="", flush=True)
                    collected.append(to_print)
                break

            # no marker yet: print newly received text
            to_print = buffer[printed_len:]
            if to_print:
                if verbose:
                    print(to_print, end="", flush=True)
                collected.append(to_print)
                printed_len = len(buffer)

    except GeneratorExit:
        pass

    thread.join()
    device_sync(device)
    total_time = time.time()
    
    if verbose:
        print()
    
    result_text = "".join(collected)
    generated_tokens = len(tokenizer.encode(result_text))
    
    prefill_time = first_token_time - prefill_start if first_token_time else 0
    decode_time = total_time - first_token_time if first_token_time else 0
    
    metrics = {
        "result": result_text,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time - prefill_start,
        "generated_tokens": generated_tokens,
        "decode_throughput": generated_tokens / decode_time if decode_time > 0 else 0,
    }
    
    return metrics


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def run_benchmark(model_path: str, device: str, use_usm: bool, max_tokens: int = 32):
    """Run benchmark with fixed input"""
    print(f"\n{'='*60}")
    print(f"Benchmark: USM={use_usm}, Device={device}, Max Tokens={max_tokens}")
    print(f"{'='*60}")
    
    set_seed(42)
    
    # Load model
    print("Loading model...")
    tokenizer, model, load_time = load_model(model_path, use_usm=use_usm, device_map=device)
    print(f"Model load time: {load_time:.4f} seconds")
    
    # Fixed user input
    user_input = "Who are you?"
    history = [f"User: {user_input}"]
    prompt = "\n".join(history) + "\nAssistant:"
    
    print(f"\nInput: {user_input}")
    print("Response: ", end="", flush=True)
    
    # Generate with measurements
    metrics = stream_generate(model, tokenizer, prompt, max_new_tokens=max_tokens, verbose=True)
    
    print("\n" + "="*60)
    print("Performance Metrics:")
    print(f"  Load time:          {load_time:.4f} s")
    print(f"  Prefill time:       {metrics['prefill_time']:.4f} s")
    print(f"  Decode time:        {metrics['decode_time']:.4f} s")
    print(f"  Total time:         {metrics['total_time']:.4f} s")
    print(f"  Generated tokens:   {metrics['generated_tokens']}")
    print(f"  Decode throughput:  {metrics['decode_throughput']:.2f} tokens/s")
    print("="*60)
    
    return {
        "usm": use_usm,
        "device": device,
        "load_time": load_time,
        "prefill_time": metrics['prefill_time'],
        "decode_time": metrics['decode_time'],
        "total_time": metrics['total_time'],
        "generated_tokens": metrics['generated_tokens'],
        "decode_throughput": metrics['decode_throughput'],
        "response": metrics['result'],
    }


def main():
    parser = argparse.ArgumentParser(description="Performance benchmark with optional USM")
    parser.add_argument("-m", "--model", required=True, help="Path to pretrained model folder")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device to use (e.g., cuda, xpu)")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max new tokens to generate")
    args = parser.parse_args()

    try:
        results = []
        
        # Run without USM first
        result_no_usm = run_benchmark(args.model, args.device, use_usm=False, max_tokens=args.max_tokens)
        results.append(result_no_usm)
        
        # Run with USM
        result_with_usm = run_benchmark(args.model, args.device, use_usm=True, max_tokens=args.max_tokens)
        results.append(result_with_usm)
        
        # Summary comparison
        print("\n" + "="*60)
        print("SUMMARY COMPARISON")
        print("="*60)
        print(f"{'Metric':<25} {'No USM':<15} {'With USM':<15}")
        print("-"*60)
        print(f"{'Load time (s)':<25} {result_no_usm['load_time']:<15.4f} {result_with_usm['load_time']:<15.4f}")
        print(f"{'Prefill time (s)':<25} {result_no_usm['prefill_time']:<15.4f} {result_with_usm['prefill_time']:<15.4f}")
        print(f"{'Decode time (s)':<25} {result_no_usm['decode_time']:<15.4f} {result_with_usm['decode_time']:<15.4f}")
        print(f"{'Total time (s)':<25} {result_no_usm['total_time']:<15.4f} {result_with_usm['total_time']:<15.4f}")
        print(f"{'Generated tokens':<25} {result_no_usm['generated_tokens']:<15} {result_with_usm['generated_tokens']:<15}")
        print(f"{'Decode throughput (tok/s)':<25} {result_no_usm['decode_throughput']:<15.2f} {result_with_usm['decode_throughput']:<15.2f}")
        print("="*60)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    