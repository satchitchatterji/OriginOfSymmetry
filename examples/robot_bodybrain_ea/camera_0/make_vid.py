import cv2
import os

# Path to the directory containing your PNG images
image_folder = '.'

# Video output file name and codec settings
video_name = 'output_video.mp4'
fps = 24  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format

# Get the list of image files
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()
print(images)

# Get the dimensions of the first image (assumes all images have the same dimensions)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Create a VideoWriter object
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Loop through the images and write them to the video
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    video.write(img)

# Release the VideoWriter
video.release()

# Close all OpenCV windows (if any)
cv2.destroyAllWindows()
