from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor
)
from qwen_vl_utils import process_vision_info
import torch
from time import time
import argparse
import yaml
import cv2
import base64
import ast
from  SmartAITool.core import *
parser = argparse.ArgumentParser()

# Load defaults from a YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser.add_argument(
    '--input',
    default=config['input_video'],
    help='path to the input video'
)
parser.add_argument(
    '--prompt',
    default=config['prompt'],
    help='user prompt on what the model should do with the input'
)
parser.add_argument(
    '--warmup_image',
    default=config['input'],  # Using the image path from config
    help='path to image used for warmup'
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
    'Qwen/Qwen2.5-VL-3B-Instruct-AWQ',
    min_pixels=256*8*8,
    max_pixels=1280*28*28,
    use_fast=True
)

# Warm up with image first
print(f"Warming up with image: {args.warmup_image}")
warmup_messages = [
    {
        'role': 'user',
        'content': [
            {
                'type': 'image',
                'image': args.warmup_image,
            },
            {'type': 'text', 'text': "Describe this image."},
        ],
    }
]

# Prepare warmup inference with image
warmup_text = processor.apply_chat_template(
    warmup_messages, tokenize=False, add_generation_prompt=True
)

warmup_image_inputs, warmup_video_inputs = process_vision_info(warmup_messages)
warmup_inputs = processor(
    text=[warmup_text],
    images=warmup_image_inputs,
    videos=warmup_video_inputs,
    padding=True,
    return_tensors='pt',
)
warmup_inputs = warmup_inputs.to('cuda')

# Log initial GPU memory usage
print("Initial GPU memory state:")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# Image warmup runs
print("\nRunning first image warmup inference...")
warmup_ids = model.generate(**warmup_inputs, max_new_tokens=128)
print("Running second image warmup inference...")
warmup_ids = model.generate(**warmup_inputs, max_new_tokens=128)

# Now prepare the actual video inference
messages = [
    {
        'role': 'user',
        'content': [
            {
                'type': 'video',
                'video': args.input,
            },
            {'type': 'text', 'text': args.prompt},
        ],
    }
]

# Preparation for video inference.
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(
    messages, 
)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors='pt',
)
inputs = inputs.to('cuda')

# Synchronize before timing
torch.cuda.synchronize()

# Start timing the actual inference
print("\nRunning timed video inference...")
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

def annotate_image(image, final_output):
    for i, output in enumerate(final_output):
        cv2.rectangle(
            image,
            pt1=(output['bbox_2d'][0], output['bbox_2d'][1]),
            pt2=(output['bbox_2d'][2], output['bbox_2d'][3]),
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            image,
            text=output['label'],
            org=(output['bbox_2d'][0], output['bbox_2d'][1]-5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
    return image

cap = cv2.VideoCapture(args.input)

# Define the video writer to save the output video.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_file = 'processed_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        img = cv2.imencode('.jpg', frame)
        encoded_image = base64.b64encode(img[1]).decode('utf-8')

        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'image': f"data:image;base64,{encoded_image}",
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

        # Inference: Generation of the output.
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        # Debugging: Print raw model output
        print(f"Raw model output:")
        cprint(output_text)  # Correctly print the list
        if not output_text or output_text[0].strip() == "[]":
            print("No humans detected in the frame.")
            final_output = []  # Default to an empty list if no output
        else:
            try:
                # Ensure string_list contains a valid Python literal
                cprint(output_text[0],'blue')
                string_list = output_text[0].strip()
                if string_list.startswith("```json"):
                    string_list = string_list[7:-3].strip()  # Strip ```json and trailing ```
                
                # Parse the cleaned JSON string using json.loads
                import json
                final_output = json.loads(string_list)
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                print(f"Error parsing model output: {e}")
                final_output = []  # Default to an empty list if parsing fails

        if not final_output:
            print("Nothing to annotate!")
        else:
            image = annotate_image(frame, final_output)
            out.write(image)  # Write the annotated frame to the output video file.
    
    else:
        break

# Release resources.
cap.release()
out.release()