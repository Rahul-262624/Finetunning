import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for Algorithmic Reasoning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Base model Hugging Face ID")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the LeetCode JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./results/llama2-algo-finetuned", help="Output directory for LoRA adapters")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    return parser.parse_args()

def format_instruction_prompt(example):
    """
    Formats the dataset into a strict prompt template optimized for 
    algorithmic reasoning and C++ code generation.
    """
    prompt = f"""### System:
You are an expert competitive programmer. Produce perfect, running C++ code that handles all edge cases. Provide the Time and Space complexity.

### Problem:
{example['problem_description']}

### Solution:
```cpp
{example['cpp_code']}
