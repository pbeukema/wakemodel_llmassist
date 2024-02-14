import click
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from dataset import WakeDataset  # Modify as necessary to import transforms

# Assuming net.py contains the EfficientNetB0KeypointDetector class
# and dataset.py contains the transforms used during training
from net import EfficientNetB0KeypointDetector


def load_model(model_path):
    """Load the trained model from a file."""
    model = EfficientNetB0KeypointDetector()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to inference mode
    return model


def preprocess_image(image_path, transform):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert(
        "L"
    )  # Assuming grayscale conversion as in your dataset
    image = transform(image)
    # Add batch dimension (BxCxHxW)
    image = image.unsqueeze(0)
    return image


def plot_keypoints(image, keypoints):
    """Plot keypoints on the image."""
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)  # Convert PIL Image to tensor if necessary

    plt.imshow(
        image.squeeze(0).cpu().numpy(), cmap="gray"
    )  # Ensure image is a tensor here
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=100, marker=".", c="cyan")
    plt.show()


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("image_path", type=click.Path(exists=True))
def run_inference(model_path, image_path):
    """Run inference on an image using a trained model."""
    # Use the same transforms as during training
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
        ]
    )

    model = load_model(model_path)
    image = preprocess_image(image_path, transform)

    # Perform inference
    with torch.no_grad():
        keypoints = model(image)
        keypoints = (
            keypoints.view(-1, 2).cpu().numpy()
        )  # Reshape and convert to numpy for plotting

    # Load original image for plotting
    original_image = Image.open(image_path).convert("L")
    plot_keypoints(original_image, keypoints)


if __name__ == "__main__":
    run_inference()
