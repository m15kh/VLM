import asyncio
import websockets
import json
import base64
import argparse
import yaml
import os
from PIL import Image
import logging
from termcolor import colored, cprint
import time
import cv2

def splitter_img(image_path):
    # Load the image
    # image = cv2.imread(image_path)
    image = image_path  # Assuming image_path is already an image array
    height, width, _ = image.shape

    # Calculate the width of each part
    part_width = width // 3

    # Draw red lines to divide the image
    cv2.line(image, (part_width, 0), (part_width, height), (0, 0, 255), 4)
    cv2.line(image, (2 * part_width, 0), (2 * part_width, height), (0, 0, 255), 4)

    # Add labels to each part
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    cv2.putText(image, "Left", (part_width // 2, 50), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(image, "Center", (part_width + part_width // 2, 50), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(image, "Right", (2 * part_width + part_width // 2, 50), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    # Save the modified image
    return image

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

# Load defaults from config
with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def image_to_base64(image_path):
    """Convert image to base64 string"""
    if not os.path.exists(image_path):
        cprint(f"Error: Image file not found: {image_path}", "red")
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    with open(image_path, "rb") as img_file:
        logger.debug(f"Reading image from {image_path}")
        return base64.b64encode(img_file.read()).decode('utf-8')

def video_frame_to_base64(video_path, frame_number=0):
    """Extract a specific frame from a video and convert it to a Base64 string."""
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        cprint(f"Error: Unable to open video file: {video_path}", "red")
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Set the frame position
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = video.read()
    if not ret:
        cprint(f"Error: Failed to read frame {frame_number} from video", "red")
        raise RuntimeError(f"Failed to read frame {frame_number} from video")
    
    # Release the video
    video.release()
    
    
    # Encode the frame to Base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    logger.debug(f"Extracted frame {frame_number}, size: {len(frame_base64)} characters")
    return frame_base64

async def connect_to_server(server_uri, camera_index, prompt, step=10):
    """Connect to the server and send every nth frame from the camera."""
    try:
        if step <= 0:
            raise ValueError("Step size must be greater than 0")
        
        logger.info(colored(f"Connecting to server at {server_uri}", "blue"))
        connection_start = time.time()
        async with websockets.connect(server_uri) as websocket:
            connection_time = time.time() - connection_start
            logger.info(colored(f"Connected to server in {connection_time:.2f}s", "green"))
            
            # Open the camera
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                cprint(f"Error: Unable to access camera with index {camera_index}", "red")
                raise RuntimeError(f"Unable to access camera with index {camera_index}")
            
            frame_number = 0
            while True:
                # Read the frame
                ret, frame = camera.read()
                if not ret:
                    cprint(f"Error: Failed to read frame {frame_number} from camera", "red")
                    continue
                
                # Process every nth frame
                if frame_number % step == 0:
                    # Encode the frame to Base64
                    frame = splitter_img(frame)
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    logger.debug(f"Extracted frame {frame_number}, size: {len(frame_base64)} characters")
                    
                    # Prepare request
                    request = {
                        "image": frame_base64,
                        "prompt": prompt,
                        "frame_number": frame_number
                    }
                    
                    # Send request
                    cprint(f"Sending frame {frame_number} with prompt: '{prompt}'", "cyan")
                    await websocket.send(json.dumps(request))
                    logger.debug(f"Frame {frame_number} sent successfully")
                    
                    # Wait for response
                    cprint("Waiting for server response...", "yellow")
                    response_start = time.time()
                    response = await websocket.recv()
                    response_time = time.time() - response_start
                    logger.info(colored(f"Response received in {response_time:.2f}s", "green"))
                    
                    result = json.loads(response)
                    
                    # Display results
                    print("\n" + "-"*50)
                    cprint(f"SERVER RESPONSE FOR FRAME {frame_number}:", "green", attrs=["bold"])
                    print("-"*50)
                    if "error" in result:
                        cprint(f"Error: {result['error']}", "red")
                    else:
                        cprint("Model output:", "green")
                        vllm_output = result['response']
                        print(vllm_output)
                        
                        folder_name = 'frames_2'
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)

                        cv2.imwrite(f'{folder_name}/frame_{frame_number}_{vllm_output}.jpg', frame)  # Save the frame for debugging

                        cprint(f"Inference time: {result['inference_time']} seconds", "blue")
                    print("-"*50)
                
                frame_number += 1
            
            # Release the camera
            camera.release()
            
    except ValueError as ve:
        cprint(f"Error: {str(ve)}", "red")
        logger.error(f"Error details: {str(ve)}")
    except websockets.exceptions.ConnectionClosedError:
        cprint("Connection to server was closed unexpectedly", "red")
        logger.error("Connection to server was closed unexpectedly")
    except Exception as e:
        cprint(f"Error: {str(e)}", "red")
        logger.error(f"Error details: {str(e)}", exc_info=True)

def main():
    logger.info(colored("Starting Qwen VL client", "blue", attrs=["bold"]))
    
    parser = argparse.ArgumentParser(description="Client for Qwen VL model inference")
    parser.add_argument(
        '--server', 
        default=f"ws://{config.get('server_host', 'localhost')}:{config.get('server_port', 8765)}",
        help='WebSocket server URI'
    )
    parser.add_argument(
        '--camera', 
        type=int, 
        default=0,
        help='Index of the camera to use (default is 0)'
    )
    parser.add_argument(
        '--prompt', 
        default=config.get('prompt', "What's in this image?"),
        help='User prompt on what the model should do with the input'
    )
    parser.add_argument(
        '--step', 
        type=int, 
        default=10,
        help='Step size for frame extraction (send every nth frame)'
    )
    
    args = parser.parse_args()
    
    # Pass the correct step argument to connect_to_server
    asyncio.run(connect_to_server(args.server, args.camera, args.prompt, args.step))

if __name__ == "__main__":
    main()
