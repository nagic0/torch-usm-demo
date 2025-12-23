import os
from pathlib import Path
import time
import threading
import argparse
import sys
from contextlib import nullcontext

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.modeling_utils import set_usm_device

ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT.parent))

from common import flush, device_sync

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


def stream_generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7):
    # conversational streaming: prompt should already include "User:" and trailing "Assistant:" marker
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

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

    prompt_len = input_ids.shape[1]
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

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # stream with detection of conversation delimiters to avoid model emitting the next "User:" turn
    buffer = ""
    printed_len = 0
    collected = []
    stop_markers = ["\nUser:", "User:", "\nAssistant:", "Assistant:"]

    try:
        for chunk in streamer:
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
                    print(to_print, end="", flush=True)
                    collected.append(to_print)
                break

            # no marker yet: print newly received text
            to_print = buffer[printed_len:]
            if to_print:
                print(to_print, end="", flush=True)
                collected.append(to_print)
                printed_len = len(buffer)

    except GeneratorExit:
        pass

    thread.join()
    print()
    return "".join(collected)


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with streaming response (optional USM)")
    parser.add_argument("-m", "--model", required=True, help="Path to pretrained model folder")
    parser.add_argument("--usm", action="store_true", help="Use USM (set USM_DEVICE=cuda)")
    parser.add_argument("--device", "-d", type=str, required=True, help="Device to use (e.g., cuda, xpu)")
    args = parser.parse_args()

    try:
        print(f"Loading model: {args.model} (USM={args.usm})")
        tokenizer, model, load_time = load_model(args.model, use_usm=args.usm, device_map=args.device)
        print(f"Loaded model in {load_time:.2f} seconds.\n")

        # simple conversation history
        history = []

        while True:
            try:
                user_input = input("User: ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            stripped = user_input.strip()
            if not stripped:
                continue

            # support quit/exit to leave the chat
            if stripped.lower() in ("quit", "exit"):
                print("\nExiting...")
                break

            # append user turn
            history.append(f"User: {stripped}")

            # build prompt from history and an Assistant marker
            prompt = "\n".join(history) + "\nAssistant:"

            print("Assistant: ", end="", flush=True)
            reply = stream_generate(model, tokenizer, prompt)

            # append assistant reply to history
            history.append(f"Assistant: {reply}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
    