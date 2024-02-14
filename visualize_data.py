import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

# Define the path to your data directory
data_dir = "ShipWakes"  # Adjust this to the path of your data directory
annotations_dir = os.path.join(data_dir, "annotations")
images_dir = os.path.join(data_dir, "imgs")

# Initialize Seaborn for better visual aesthetics
sns.set(style="whitegrid", palette="muted")

# Create a list to hold file paths for images and their corresponding annotations
image_files = []
annotation_files = []

# Loop through the annotations directory to get the list of annotation files
for annotation_file in os.listdir(annotations_dir):
    if annotation_file.endswith(".json"):
        annotation_path = os.path.join(annotations_dir, annotation_file)
        image_file = annotation_file.replace(
            ".json", ".png"
        )  # Assuming image file names match annotation file names
        image_path = os.path.join(images_dir, image_file)

        # Check if the corresponding image file exists
        if os.path.exists(image_path):
            annotation_files.append(annotation_path)
            image_files.append(image_path)

# Plotting
num_examples = min(len(image_files), 10)  # Limiting to 10 examples for visualization
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for idx, (image_path, annotation_path) in enumerate(
    zip(image_files[:num_examples], annotation_files[:num_examples])
):
    # Load the image
    img = Image.open(image_path).convert("RGB")  # Ensure the image is treated as RGB
    draw = ImageDraw.Draw(img)

    # Load the corresponding annotations and draw keypoints
    with open(annotation_path, "r") as f:
        annotations = json.load(f)
        for point in annotations["tooltips"]:
            x, y = point["x"], point["y"]
            # Draw keypoints in cyan for visibility
            draw.ellipse(
                [(x - 10, y - 10), (x + 10, y + 10)], fill="cyan", outline="black"
            )

    # Plot the image with keypoints
    ax = axes[idx // 5, idx % 5]
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Image {idx+1}")

plt.tight_layout()
plt.show()
