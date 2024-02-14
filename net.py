from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetB0KeypointDetector(nn.Module):
    """
    A custom EfficientNet-B0 model for keypoint detection in grayscale images.
    The network modifies the input layer to accept single-channel images and
    adds a custom output layer for predicting a fixed number of keypoints.
    """

    def __init__(self, num_keypoints: int = 3):
        """
        Initializes the model.

        Parameters:
        - num_keypoints: The number of keypoints to predict. Default is 3,
                         corresponding to the maximum number of keypoints in the dataset.
        """
        super(EfficientNetB0KeypointDetector, self).__init__()
        self.num_keypoints = num_keypoints
        # Load a pre-trained EfficientNet-B0
        self.efficientnet_b0 = models.efficientnet_b0(pretrained=True)

        # Modify the first convolutional layer to accept single-channel (grayscale) images
        original_first_conv = self.efficientnet_b0.features[0][0]
        self.efficientnet_b0.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=False,
        )

        # Modify the classifier to predict keypoints (num_keypoints * 2 because x, y for each keypoint)
        self.efficientnet_b0.classifier = nn.Linear(
            in_features=self.efficientnet_b0.classifier[1].in_features,
            out_features=num_keypoints * 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters:
        - x: Input tensor of shape (batch_size, 1, 224, 224)

        Returns:
        - A tensor of shape (batch_size, num_keypoints * 2) representing the predicted keypoints.
        """
        return self.efficientnet_b0(x)


# Example usage
if __name__ == "__main__":
    model = EfficientNetB0KeypointDetector()
    print(model)

    # Example input tensor (batch_size, channels, height, width)
    example_input = torch.randn(1, 1, 224, 224)
    output = model(example_input)
    print(
        output.shape
    )  # Expected shape: (1, 6) for 3 keypoints (each with x, y coordinates)
