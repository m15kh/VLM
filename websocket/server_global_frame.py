import asyncio
import websockets
import json
import base64
import torch
import yaml
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from time import time
import os
import logging
from termcolor import colored, cprint
import time as tm

# Set up colorized logging
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }

    def format(self, record):
        log_message = super().format(record)
        return colored(log_message, self.COLORS.get(record.levelname, 'white'))

# Configure logger with color
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Remove existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create console handler with color formatting
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Also create a file handler for persistent logs
log_file_path = 'server.log'
# Clear the log file before starting the server
with open(log_file_path, 'w'):
    pass

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Load config
with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)


class QwenInferenceServer:
    def __init__(self):
        cprint("Initializing Qwen VL Inference Server", "blue", attrs=["bold"])
        logger.info("Initializing Qwen VL model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cprint(f"Using device: {self.device}", "yellow")
        
        # Load model
        start_time = tm.time()
        cprint("Loading model Qwen/Qwen2.5-VL-7B-Instruct...", "cyan") #Qwen/Qwen2.5-VL-7B-Instruct
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
        
    def _warmup(self):
        # Create a sample input for warmup
        sample_img_path = config.get('sample_warmup_image', 'sample_image.jpg')
        if not os.path.exists(sample_img_path):
            logger.warning(colored(f"Sample image {sample_img_path} not found. Creating a dummy image.", "yellow"))
            # Create a dummy image if sample not available
            import numpy as np
            from PIL import Image
            dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            dummy_img.save(sample_img_path)
            cprint(f"Created dummy image at {sample_img_path}", "yellow")
        
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': sample_img_path},
                    {'type': 'text', 'text': "What's in this image?"}
                ]
            }
        ]
        
        cprint("Preparing warmup inputs...", "cyan")
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
        ).to(self.device)
        
        cprint("Running first warmup inference...", "yellow")
        start = tm.time()
        _ = self.model.generate(**inputs, max_new_tokens=128)
        first_warmup_time = tm.time() - start
        cprint(f"First warmup completed in {first_warmup_time:.2f} seconds", "green")
        
        cprint("Running second warmup inference...", "yellow")
        start = tm.time()
        _ = self.model.generate(**inputs, max_new_tokens=128)
        second_warmup_time = tm.time() - start
        cprint(f"Second warmup completed in {second_warmup_time:.2f} seconds", "green")
        
        cprint("Warmup completed successfully", "green", attrs=["bold"])
        
        # Log memory usage
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        cprint(f"GPU memory allocated: {allocated:.2f} MB", "blue")
        cprint(f"GPU memory reserved: {reserved:.2f} MB", "blue")
        logger.info(f"GPU memory stats - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

    async def process_request(self, image_data, prompt):
        request_id = f"req_{tm.time():.6f}"
        logger.info(colored(f"Processing request {request_id} with prompt: '{prompt[:30]}...'", "cyan"))
        
        # Save the received image temporarily
        temp_img_path = f"temp_received_image_{request_id}.jpg"
        try:
            with open(temp_img_path, "wb") as f:
                f.write(base64.b64decode(image_data))
            logger.debug(f"Saved temporary image to {temp_img_path}")
            
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': temp_img_path},
                        {'type': 'text', 'text': prompt}
                    ]
                }
            ]
            
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
            ).to(self.device)
            
            # Run inference
            cprint(f"[{request_id}] Starting inference...", "yellow")
            torch.cuda.synchronize()
            t1 = time()
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            torch.cuda.synchronize()
            inference_time = time() - t1
            cprint(f"[{request_id}] Inference completed in {inference_time:.2f} seconds", "green")
            
            # Process output
            cprint(f"[{request_id}] Processing output...", "cyan")
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Log memory usage after inference
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.debug(f"GPU memory after inference - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
            
            return {
                "response": output_text[0],
                "inference_time": f"{inference_time:.2f}",
                "request_id": request_id
            }
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                logger.debug(f"Removed temporary image {temp_img_path}")

async def handle_client(websocket, server):
    client_id = f"client_{tm.time():.6f}"
    try:
        cprint(f"Client {client_id} connected", "green")
        logger.info(f"Client {client_id} connected")
        
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if "image" in data and "prompt" in data:
                    image_data = data["image"]  # Base64 encoded image
                    prompt = data["prompt"]
                    
                    img_size = len(image_data) / 1024
                    cprint(f"Received request from {client_id}", "cyan")
                    logger.info(f"Image size: {img_size:.2f} KB, Prompt length: {len(prompt)}")
                    
                    # Process the request
                    cprint(f"Processing request from {client_id}...", "yellow")
                    result = await server.process_request(image_data, prompt)
                    
                    # Add client ID to response
                    result["client_id"] = client_id
                    
                    # Send response back
                    cprint(f"Sending response to {client_id}...", "green")
                    await websocket.send(json.dumps(result))
                    logger.info(f"Response sent to {client_id}")
                else:
                    cprint(f"Invalid request format from {client_id}", "red")
                    await websocket.send(json.dumps({"error": "Invalid request format"}))
                    
            except json.JSONDecodeError:
                cprint(f"Invalid JSON received from {client_id}", "red")
                await websocket.send(json.dumps({"error": "Invalid JSON format"}))
                
    except websockets.exceptions.ConnectionClosed:
        cprint(f"Client {client_id} disconnected", "yellow")
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        cprint(f"Error handling client {client_id}: {str(e)}", "red")
        logger.error(f"Error handling client {client_id}: {str(e)}", exc_info=True)

async def start_server():
    server = QwenInferenceServer()
    
    host = config.get('server_host', 'localhost')
    port = config.get('server_port', 8765)
    
    cprint(f"Starting WebSocket server on {host}:{port}...", "blue", attrs=["bold"])
    
    async with websockets.serve(
        lambda ws: handle_client(ws, server), 
        host, 
        port
    ):
        cprint(f"Server started at ws://{host}:{port}", "green", attrs=["bold"])
        logger.info(f"Server started at ws://{host}:{port}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        cprint("Starting Qwen VL Inference Server", "green", attrs=["bold"])
        asyncio.run(start_server())
    except KeyboardInterrupt:
        cprint("Server shutting down...", "yellow")
    except Exception as e:
        cprint(f"Fatal error: {str(e)}", "red")
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
