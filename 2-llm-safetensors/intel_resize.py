import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

max_shared_size = 4 * 1024 * 1024 * 1024  # 4GB

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, required=True, help="Path to load the model from")
    parser.add_argument("--save", type=str, required=True, help="Path to save the resized model to")
    args = parser.parse_args()

    load_path = args.load
    save_path = args.save

    # Load
    print(f"Loading model from {load_path}...")
    model = AutoModelForCausalLM.from_pretrained(load_path, dtype="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(load_path)

    # Save
    model.save_pretrained(save_path, max_shard_size=max_shared_size)
    tokenizer.save_pretrained(save_path)
    print(f"Resized model saved to {save_path}.")