# Wake Detection Library

## TL;DR

This library provides an end-to-end solution for training and inferring wake detection in satellite imagery (Sentinel-2) using deep learning. It utilizes an EfficientNet-B0 architecture adapted for keypoint detection, specifically identifying keypoints in images that represent vessel wakes. The library is structured into modular components for ease of use, including data handling, model definition, training, evaluation, and inference.

## Modules

### `net.py`

Defines the `EfficientNetB0KeypointDetector` class, a neural network model based on the EfficientNet-B0 architecture adapted for keypoint detection tasks. It outputs coordinates for a predefined number of keypoints representing vessel wakes.

### `dataset.py`

Implements the `WakeDataset` class, a custom PyTorch dataset for loading and preprocessing satellite imagery and corresponding keypoint annotations. It includes functionality to apply transformations to images and filter annotations based on the number of keypoints.

### `traineval.py`

Contains functions for training and evaluating the model on the wake detection task. It supports training the model with gradient descent, evaluating its performance on a validation set, and logging metrics to Weights & Biases.

### `inference.py`

Provides functionality for loading a trained model and performing keypoint detection on new images. It also includes a command-line interface for easy inference on individual images.

## Commands

### Training

To train the model, ensure you have a dataset organized according to the expected directory structure and execute the following command:

```bash
python traineval.py --data-dir /path/to/dataset
```

Replace `/path/to/dataset` with the actual path to your dataset. note that the dataset can be downloaded from:

### Inference

For running inference with a trained model on a new image:

```bash
python inference.py /path/to/model.pth /path/to/image.png
```

Ensure you replace `/path/to/model.pth` with the path to your trained model file and `/path/to/image.png` with the path to the image you want to process. A trained model is provided in this repository (see trained_models/)

## Important Assumptions

- **Dataset Structure**: The library assumes that the dataset is organized into separate directories for images and annotations, with annotations in JSON format specifying keypoints as x, y coordinates.
- **Preprocessing**: Images are resized to 224x224 pixels and converted to grayscale before being fed into the model. This preprocessing step is crucial for the model to correctly interpret the input data.
- **Keypoint Representation**: It's assumed that each annotation can have between 2 to 3 keypoints. Annotations with fewer than 2 keypoints are filtered out during data loading.
- **Model Output**: The model is designed to output a fixed number of keypoints (up to 3), represented as flat x, y coordinates. This requires the inference process to interpret the model's output accordingly.

---
