import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import wandb
from dataset import WakeDataset  # Assuming dataset.py is in the same directory
from net import EfficientNetB0KeypointDetector


def custom_collate_fn(batch):
    images = [item["image"] for item in batch]
    keypoints = [item["keypoints"] for item in batch]
    images = torch.stack(images, 0)
    keypoints = torch.stack(keypoints, 0)
    return images, keypoints


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    model_save_path: str,
):
    """
    Trains the model for one epoch.

    Parameters:
    - model: The neural network model.
    - dataloader: DataLoader providing the training data.
    - optimizer: Optimizer used for model training.
    - device: The device to train on.
    """
    model.train()
    total_loss = 0.0
    for images, keypoints in dataloader:
        print(total_loss)

        images, keypoints = images.to(device), keypoints.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # loss = nn.MSELoss()(outputs, keypoints)
        loss = nn.MSELoss()(outputs, keypoints)

        r_loss = torch.sqrt(loss)
        r_loss.backward()
        optimizer.step()
        total_loss += r_loss.item()

    average_loss = total_loss / len(dataloader)
    wandb.log({"train_loss": average_loss})
    # Save the model checkpoint
    model_filename = f"model_epoch_{epoch}.pth"
    model_save_path_full = os.path.join(model_save_path, model_filename)
    torch.save(model.state_dict(), model_save_path_full)
    print(f"Model saved to {model_save_path_full}")


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Evaluates the model on the validation set.

    Parameters:
    - model: The neural network model.
    - dataloader: DataLoader providing the validation data.
    - device: The device to evaluate on.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, keypoints in dataloader:
            print("val_loss " + str(total_loss))
            images, keypoints = images.to(device), keypoints.to(device)

            outputs = model(images)

            loss = nn.MSELoss()(outputs, keypoints)

            r_loss = torch.sqrt(loss)
            total_loss += r_loss.item()

    average_loss = total_loss / len(dataloader)
    wandb.log({"val_loss": average_loss})


def main():
    # Initialize Weights & Biases
    wandb.init(project="wake_model_llm_assist")

    # Setup
    if torch.backends.mps.is_available():  # Check if MPS backend is available
        print("Using MPS backend for acceleration on Apple Silicon.")
        device = torch.device("mps")  # Use MPS device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetB0KeypointDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Load dataset
    dataset = WakeDataset(
        data_dir="ShipWakes",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
            ]
        ),
    )

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=1,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=1,
        pin_memory=True,
    )
    model_save_path = "./model_checkpoints"  # Define your model save directory
    # Training loop
    num_epochs = 500  # Define the number of epochs
    for epoch in range(num_epochs):

        train_one_epoch(
            model, train_dataloader, optimizer, device, epoch, model_save_path
        )
        evaluate(model, val_dataloader, device)
        # Log additional metrics or images if needed
    wandb.finish()


if __name__ == "__main__":
    main()
