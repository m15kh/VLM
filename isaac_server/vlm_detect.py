from termcolor import cprint
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import logging
import time as tm
from qwen_vl_utils import process_vision_info
from time import time
import os
from io import BytesIO
from PIL import Image


logger = logging.getLogger(__name__)

class QwenInferenceServer:
    def __init__(self):
        cprint("Initializing Qwen VL Inference Server", "blue", attrs=["bold"])
        logger.info("Initializing Qwen VL model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cprint(f"Using device: {self.device}", "yellow")
        
        # Load model
        start_time = tm.time()
        cprint("Loading model Qwen/Qwen2.5-VL-7B-Instruct...", "cyan")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            'Qwen/Qwen2.5-VL-7B-Instruct', 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        model_load_time = tm.time() - start_time
        cprint(f"Model loaded in {model_load_time:.2f} seconds", "green")
        
        # Load processor
        proc_start_time = tm.time()
        cprint("Loading processor...", "cyan")
        self.processor = AutoProcessor.from_pretrained(
            'Qwen/Qwen2.5-VL-7B-Instruct',
            min_pixels=256*8*8,
            max_pixels=1280*28*28,
            use_fast=True
        )
        proc_load_time = tm.time() - proc_start_time
        cprint(f"Processor loaded in {proc_load_time:.2f} seconds", "green")
        
        # Perform model warmup
        cprint("Starting model warmup...", "yellow", attrs=["bold"])
        self._warmup()
        
        cprint("Server initialized successfully", "green", attrs=["bold"])

    def analyze_image(self, image_input, prompt="What do you see in this image? Describe it in detail."):
        """Analyze an image using the Qwen model."""
        try:
            # Handle both file paths and BytesIO inputs
            if isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
            else:
                image = Image.open(image_input).convert('RGB')

            messages = [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'image': image,
                        },
                        {'type': 'text', 'text': prompt},
                    ],
                }
            ]

            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors='pt',
            )
            inputs = inputs.to(self.device)

            # Generate response
            t1 = time()
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            torch.cuda.synchronize()
            inference_time = time() - t1

            # Process output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0], inference_time
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}", 0

    def _warmup(self):
        """Warmup the model with a test image."""
        try:
            test_img_path = "/home/ubuntu7/m15kh/vllm/isaac_server/data/test.jpg"
            if not os.path.exists(test_img_path):
                logger.warning(f"Test image not found at {test_img_path}")
                return

            cprint("\nInitial GPU memory state:", "yellow")
            cprint(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB", "yellow")
            cprint(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB", "yellow")

            # First warmup run
            cprint("\nRunning first warmup inference...", "cyan")
            result, _ = self.analyze_image(test_img_path)
            
            # Second warmup run
            cprint("Running second warmup inference...", "cyan")
            result, inference_time = self.analyze_image(test_img_path)
            
            cprint("\nWarmup analysis result:", "green")
            cprint(result, "green")
            cprint(f"\nInference time (after warmup): {inference_time:.2f} seconds", "yellow")
            
            # Log final GPU state
            cprint(f"\nGPU memory allocated after warmup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB", "yellow")
            cprint(f"GPU memory reserved after warmup: {torch.cuda.memory_reserved() / 1024**2:.2f} MB", "yellow")
            
        except Exception as e:
            logger.error(f"Error during warmup: {str(e)}")
            cprint(f"Warmup failed: {str(e)}", "red")

