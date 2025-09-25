import asyncio
import websockets
import json
import base64
from colorama import init, Fore, Style
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import io
import time

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

SERVER_URL = "-"

class VLMRobotController(Node):
    def __init__(self):
        super().__init__('vlm_robot_controller')
        self.bridge = CvBridge()
        self.last_capture_time = 0
        self.cmd_vel_publisher = self.create_publisher(Twist, '/husky_controller/cmd_vel', 10)
        
        # Create subscription to the camera topic
        self.subscription = self.create_subscription(
            Image,
            '/front_stereo_camera/front_rgb/image_raw',
            self.image_callback,
            10
        )
        self.ws_client = None
        
    async def connect_websocket(self):
        try:
            self.ws_client = await websockets.connect(SERVER_URL)
            Logger.success("Connected to WebSocket server")
            return True
        except Exception as e:
            Logger.error(f"Failed to connect to WebSocket server: {e}")
            return False

    def publish_movement(self, linear_x=0.0, angular_z=0.0):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_vel_publisher.publish(twist)

    async def process_vlm_response(self, response_data):
        command = response_data['analysis']
        if command == "Move_Forward":
            Logger.info("Moving forward")
            self.publish_movement(linear_x=0.8, angular_z=0.0)
        elif command == "Turn_Left":
            Logger.info("Turning left")
            self.publish_movement(linear_x=0.0, angular_z=0.8)
        elif command == "Turn_Right":
            Logger.info("Turning right")
            self.publish_movement(linear_x=0.0, angular_z=-0.8)

    async def send_image_to_vlm(self, cv_image):
        if self.ws_client is None:
            await self.connect_websocket()
        
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', cv_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        try:
            message = {"image": base64_image}
            await self.ws_client.send(json.dumps(message))
            response = await self.ws_client.recv()
            response_data = json.loads(response)
            await self.process_vlm_response(response_data)
        except Exception as e:
            Logger.error(f"Error in VLM processing: {e}")
            self.ws_client = None

    async def image_callback(self, msg):
        current_time = time.time()
        # Changed from 1 second to 0.5 seconds for 2 frames per second
        if current_time - self.last_capture_time >= 0.5:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                h, w, _ = cv_image.shape
                third_w = w // 3
                
                # Split image into three regions
                left_img = cv_image[:, :third_w]
                center_img = cv_image[:, third_w:2*third_w]
                right_img = cv_image[:, 2*third_w:]
                
                # Process each region with VLM
                await self.send_image_to_vlm(cv_image)  # Send full image
                
                self.last_capture_time = current_time
                
            except Exception as e:
                Logger.error(f"Error processing image: {e}")

async def main():
    rclpy.init()
    controller = VLMRobotController()
    
    while True:
        rclpy.spin_once(controller, timeout_sec=0.1)
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())
