import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class WakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform  # Use the passed transform
        self.annotations_dir = os.path.join(data_dir, "annotations")
        self.images_dir = os.path.join(data_dir, "imgs")
        self.image_files = []
        self.annotation_files = []

        # Load and filter annotations
        for annotation_file in os.listdir(self.annotations_dir):
            if annotation_file.endswith(".json"):
                annotation_path = os.path.join(self.annotations_dir, annotation_file)
                with open(annotation_path, "r") as f:
                    annotations = json.load(f)
                    # Ensure there are at least 2 keypoints
                    if (
                        # len(annotations["tooltips"]) >= 2
                        len(annotations["tooltips"])
                        == 3
                    ):
                        image_file = annotation_file.replace(".json", ".png")
                        image_path = os.path.join(self.images_dir, image_file)
                        # Check if the corresponding image file exists
                        if os.path.exists(image_path):
                            self.annotation_files.append(annotation_path)
                            self.image_files.append(image_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        annotation_path = self.annotation_files[idx]
        # Load image
        image = Image.open(image_path).convert("L")  # Convert grayscale images to RGB
        # Load annotations
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
            keypoints_list = [
                torch.tensor(list(point.values()), dtype=torch.float)
                for point in annotations["tooltips"]
            ]

        keypoints_flat = torch.zeros(
            6
        )  # Initialize a zero tensor for 3 keypoints (x, y)
        for i, kp in enumerate(keypoints_list):
            keypoints_flat[2 * i : 2 * (i + 1)] = kp  # Fill in the keypoints

        # Check if transform is provided
        if self.transform:
            image = self.transform(image)
        else:
            # Apply default transform if none provided
            default_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),  # Resizing the image
                    transforms.ToTensor(),  # Convert the PIL Image to a tensor
                ]
            )
            image = default_transform(image)

        return {"image": image, "keypoints": keypoints_flat}
