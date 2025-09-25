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
import ast



def splitter_img(image):
    height, width, _ = image.shape

    part_width = width // 3

    cv2.line(image, (part_width, 0), (part_width, height), (0, 0, 255), 4)
    cv2.line(image, (2 * part_width, 0), (2 * part_width, height), (0, 0, 255), 4)

    # Add labels to each part
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    cv2.putText(image, "Left", (part_width // 2, 50), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(image, "Center", (part_width + part_width // 2 - 40, 40 ), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
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
    """Convert image to base64 string after processing"""
    frame = cv2.imread(image_path)
    frame = splitter_img(frame)       # preprocess
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

async def connect_to_server(server_uri, image_path, prompt):
    """Connect to the server and send a request"""
    try:
        logger.info(colored(f"Connecting to server at {server_uri}", "blue"))
        connection_start = time.time()
        async with websockets.connect(server_uri) as websocket:
            connection_time = time.time() - connection_start
            logger.info(colored(f"Connected to server in {connection_time:.2f}s", "green"))
            
            # Convert image to base64
            
            if not os.path.exists(image_path):
                cprint(f"Error: Image file not found: {image_path}", "red")
                raise FileNotFoundError(f"Image file not found: {image_path}")

            #DEBUGGER
            # frame = cv2.imread(image_path)
            # cv2.imwrite('split_image.jpg', frame)
            # cprint("Split image saved as 'split_image.jpg'", "green")
            #DEBUGGER
            cprint("Converting image to base64...", "cyan")

            image_data = image_to_base64(image_path)
            logger.debug(f"Image converted, size: {len(image_data)} characters")
            
            # Prepare request
            request = {
                "image": image_data,
                "prompt": prompt
            }
            
            # Send request
            cprint(f"Sending request with prompt: '{prompt}'", "cyan")
            await websocket.send(json.dumps(request))
            logger.debug("Request sent successfully")
            
            # Wait for response
            cprint("Waiting for server response...", "yellow")
            response_start = time.time()
            response = await websocket.recv()
            response_time = time.time() - response_start
            logger.info(colored(f"Response received in {response_time:.2f}s", "green"))
            
            result = json.loads(response)
            # Display results
            print("\n" + "-"*50)
            cprint("SERVER RESPONSE:", "green", attrs=["bold"])
            print("-"*50)
            if "error" in result:
                cprint(f"Error: {result['error']}", "red")
            else:
                cprint("Model output:", "green")
                
                #@m15kh
                print(result['response'])
                
                cprint('---------------------#######------------------','red')
                
                cprint(f"Inference time: {result['inference_time']} seconds", "blue")
                
            print("-"*50)
            
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
        '--image', 
        default=config['input'],
        help='Path to the input image'
    )
    parser.add_argument(
        '--prompt', 
        default=config.get('prompt', "What's in this image?"),
        help='User prompt on what the model should do with the input'
    )
    
    args = parser.parse_args()
    
    # Run the client
    asyncio.run(connect_to_server(args.server, args.image, args.prompt))

if __name__ == "__main__":
    main()
