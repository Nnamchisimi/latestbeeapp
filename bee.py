import torch
from pathlib import Path
import cv2
from tqdm import tqdm  # Optional, for progress bars

# Define paths
model_path = r'C:\Users\CIU\Desktop\beemodel\best.pt'
image_directory = r'C:\Users\CIU\Desktop\beemodel\images'
output_directory = r'C:\Users\CIU\Desktop\beeapp\static\detections'  # Output directory for saving annotated images

# Ensure output directory exists
Path(output_directory).mkdir(parents=True, exist_ok=True)

# Load model
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    print("YOLOv5 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    exit()

# Set model to evaluation mode
model.eval()

# Get list of image files in the image directory
image_files = list(Path(image_directory).glob('*.jpg'))

print(f"Found {len(image_files)} image(s) in {image_directory}")

# Create a window to display images
cv2.namedWindow('Detection Results', cv2.WINDOW_NORMAL)

# Process each image
for img_path in tqdm(image_files, desc='Processing images'):
    img_name = img_path.stem
    print(f"Processing image: {img_path}")

    img = cv2.imread(str(img_path))[:, :, ::-1]  # Read image with OpenCV (BGR to RGB)
    
    try:
        # Inference
        results = model(img)
        
        # Get annotated image from results
        annotated_img = results.render()[0]  # Get the first image with detections
        
        # Save annotated image to output directory
        save_path = Path(output_directory) / f'{img_name}.jpg'
        cv2.imwrite(str(save_path), annotated_img[:, :, ::-1])  # Save as BGR (OpenCV's default)
        
        # Display annotated image
        cv2.imshow('Detection Results', annotated_img)
        
        # Wait for a key press and check if 'q' is pressed to exit
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Close OpenCV windows
cv2.destroyAllWindows()

print(f"Detected images saved in: {output_directory}")
