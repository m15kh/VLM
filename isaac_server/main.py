import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from termcolor import cprint
import time as tm
from time import time
import os
import logging
import json
from starlette.websockets import WebSocketDisconnect
from PIL import Image
import uvicorn
from vlm_detect import QwenInferenceServer
import yaml
import base64
from io import BytesIO
from colorama import init, Fore, Style

# Initialize colorama
init()

class Logger:
    @staticmethod
    def success(msg):
        print(f"{Fore.GREEN}[SUCCESS] {msg}{Style.RESET_ALL}")
    
    @staticmethod
    def error(msg):
        print(f"{Fore.RED}[ERROR] {msg}{Style.RESET_ALL}")
    
    @staticmethod
    def info(msg):
        print(f"{Fore.BLUE}[INFO] {msg}{Style.RESET_ALL}")
    
    @staticmethod
    def warning(msg):
        print(f"{Fore.YELLOW}[WARNING] {msg}{Style.RESET_ALL}")

# Configure logging
logger = logging.getLogger(__name__)

# Global variable to store model instance
qwen_server = None





@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the model in the lifespan
    global qwen_server
    qwen_server = QwenInferenceServer()
    yield
    # Cleanup (if needed)
    qwen_server = None

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok"}



# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

DEBUG_FOLDER = "debug_images"
if config.get('debug', {}).get('save_images', False):
    os.makedirs(DEBUG_FOLDER, exist_ok=True)
    Logger.info(f"Debug mode enabled - saving images to {DEBUG_FOLDER}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        Logger.success("Client connected successfully")
        frame_count = 0
        
        while True:
            try:
                message = await websocket.receive()
                
                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if "image" in data:
                            frame_count += 1
                            Logger.info(f"Processing frame {frame_count}")
                            # Decode base64 image data
                            image_bytes = base64.b64decode(data["image"])
                            frame_data = BytesIO(image_bytes)
                            
                            # Save image if debug mode is enabled
                            if config.get('debug', {}).get('save_images', False):
                                timestamp = tm.strftime("%Y%m%d-%H%M%S")
                                debug_filename = f"{DEBUG_FOLDER}/frame_{frame_count}_{timestamp}.jpg"
                                with open(debug_filename, "wb") as f:
                                    f.write(image_bytes)
                                Logger.info(f"Saved debug image: {debug_filename}")
                            
                            # Analyze frame with VLM directly from memory
                            Logger.info("Starting VLM inference...")
                            start_time = time()
                            analysis_result, inference_time = qwen_server.analyze_image(
                                frame_data,
                                prompt=config['vlm']['prompt']
                            )
                            
                            Logger.success(f"VLM Output: {analysis_result}")
                            Logger.info(f"Inference time: {inference_time:.2f}s")
                            
                            # Send analysis results back to client
                            await websocket.send_json({
                                "status": "analyzed",
                                "frame_id": frame_count,
                                "analysis": analysis_result,
                                "inference_time": inference_time
                            })
                        else:
                            Logger.warning("Received message without image data")
                            await websocket.send_json({"status": "received", "type": "json"})
                    except json.JSONDecodeError:
                        Logger.error("Invalid JSON received")
                        await websocket.send_json({"status": "error", "message": "Invalid JSON"})
                    except Exception as e:
                        Logger.error(f"Processing error: {str(e)}")
                        await websocket.send_json({"status": "error", "message": str(e)})
                        
            except WebSocketDisconnect:
                Logger.info("Client disconnected")
                break
                
    except WebSocketDisconnect:
        Logger.info("Client disconnected during handshake")
    except Exception as e:
        Logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(code=1001)
        except:
            pass  # Ignore errors during forced closure





if __name__ == "__main__":
    # Use module:app format instead of passing app directly
    uvicorn.run("main:app", host="0.0.0.0", port=20807, reload=True)