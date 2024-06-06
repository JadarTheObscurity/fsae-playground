import cv2
import os
import numpy as np

def create_video(folder_path, video_name, fps=30):
    images = []
    # Check if file not found error
    if not os.path.exists(folder_path):
        print(f"[!] Folder not found: {folder_path}")
        return
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            images.append(image)

    # Determine the width and height from the first image
    height, width, layers = images[0].shape

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    video_path = os.path.join(folder_path, video_name)
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Add images to video
    for image in images:
        video.write(image)

    # Release the video writer
    video.release()

# Example usage
folder_path = r"./Curve Fitting/__tmp_pic"
video_name = "animation.mp4"
create_video(folder_path, video_name, fps=5)