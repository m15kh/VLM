import asyncio
import websockets
import json
from PIL import Image
import io
import base64

SERVER_URL = "-"
ws_client = None

async def connect_websocket():
    global ws_client
    try:
        ws_client = await websockets.connect(SERVER_URL)
        print("Connected to WebSocket server")
        return True
    except Exception as e:
        print(f"Failed to connect to WebSocket server: {e}")
        return False

async def send_frame(image_path):
    global ws_client
    if ws_client is None:
        await connect_websocket()
    
    try:
        # Read and encode image to base64
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create JSON message
        message = {
            "image": base64_image
        }
        
        # Send JSON message
        await ws_client.send(json.dumps(message))
        
        # Receive and parse response
        response = await ws_client.recv()
        response_data = json.loads(response)
        
        print(f"Received analysis result:")
        print(f"Frame ID: {response_data['frame_id']}")
        print(f"Analysis: {response_data['analysis']}")
        print(f"Inference time: {response_data['inference_time']:.2f}s")
        
    except Exception as e:
        print(f"Error sending frame: {e}")
        ws_client = None  # Reset connection on error

async def close_connection():
    global ws_client
    if ws_client:
        await ws_client.close()
        ws_client = None

async def main():
    # Connect to server
    await connect_websocket()
    
    # Send your frame
    await send_frame("1.png")  # Replace with your image path
    
    # When done, close connection
    await close_connection()

if __name__ == "__main__":
    asyncio.run(main())
