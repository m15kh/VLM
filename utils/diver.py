import cv2
import numpy as np

def splitter_img(image_path):
    # Load the image
    image = cv2.imread(image_path)
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

# Example usage
out = splitter_img("/home/ubuntu7/m15kh/vllm/Qwen_Inference/websocket/fr/frame_230_Turn Left.jpg")
cv2.imwrite("output_image.jpg", out)
