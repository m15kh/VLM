from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor
)
from qwen_vl_utils import process_vision_info
import torch
import argparse

from time import time
import yaml


parser = argparse.ArgumentParser()


# Load defaults from a YAML file
with open('/home/ubuntu7/m15kh/vllm/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser.add_argument(
    '--input',
    default=config['input'],
    help='path to the input image(s)'
)

parser.add_argument(
    '--prompt',
    default=config['prompt'],
    help='user prompt on what the model should do with the input'
)
args = parser.parse_args()

# Load model.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct', 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# Load processor.
processor = AutoProcessor.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct',
    min_pixels=256*8*8,
    max_pixels=1280*28*28,
    use_fast=True
    
)

messages = [
    {
        'role': 'user',
        'content': [
            {
                'type': 'image',
                'image': args.input,
            },
            {'type': 'text', 'text': args.prompt},
        ],
    }
]

# Preparation for inference.
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors='pt',
)
inputs = inputs.to('cuda')

# Log initial GPU memory usage
print("Initial GPU memory state:")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# First warmup run
print("\nRunning first warmup inference...")
warmup_ids = model.generate(**inputs, max_new_tokens=128)

# Second warmup run
print("Running second warmup inference...")
warmup_ids = model.generate(**inputs, max_new_tokens=128)

# Synchronize before timing
torch.cuda.synchronize()

# Start timing the actual inference
print("\nRunning timed inference...")
t1 = time()
generated_ids = model.generate(**inputs, max_new_tokens=128)
torch.cuda.synchronize()  # Wait for GPU operations to finish
inference_time = time() - t1

# Log GPU memory usage after inference.
print(f"\nGPU memory allocated after inference: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"GPU memory reserved after inference: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)
print("\nModel output:")
print(output_text[0])

print(f"\nInference time (after warmup): {inference_time:.2f} seconds")