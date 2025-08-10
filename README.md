Here is the text you selected from the Canvas.

# MNIST Handwritten Digits Classification

This repository contains a Jupyter Notebook that demonstrates the process of building, training, and evaluating a neural network for classifying handwritten digits from the MNIST dataset.

## Project Introduction

The MNIST dataset is a classic "hello world" for image classification. It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). This project aims to build a custom neural network model to achieve high accuracy on this dataset. The notebook guides you through:

  * Loading and preprocessing the MNIST dataset using `torchvision`.

  * Justifying the chosen preprocessing steps.

  * Exploring the dataset to understand its structure.

  * Constructing a neural network with `torch.nn`.

  * Training the model and tracking its performance.

  * Evaluating the model's accuracy on the test set.

  * Improving the model architecture to boost performance.

  * Saving the trained model weights.

## Installation

To run the notebook, you need to have Python and the required libraries installed. It's recommended to use a virtual environment.

First, clone this repository:

```
git clone <repository_url>
cd <repository_name>

```

Then, install the dependencies listed in `requirements.txt`:

```
pip install -r requirements.txt

```

You may also need to install PyTorch separately if your environment requires a specific version for GPU support. Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command.

## Model Architecture

The initial model, `MNISTNet`, is a simple fully-connected neural network.

The improved model, `CNNNet`, is a Convolutional Neural Network (CNN) designed to better capture spatial features in the images.

```
import torch.nn as nn
import torch.nn.functional as F

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

```

## Results

After training the `CNNNet` model for 5 epochs with the Adam optimizer and CrossEntropyLoss, the model achieved a test accuracy of approximately **97.17%**.

The training loss curve shows a steady decrease over the epochs, indicating that the model is learning effectively.

## Usage

1.  **Open the Notebook:** Launch Jupyter Notebook or JupyterLab and open the `MNIST_Handwritten_Digits-STARTER.ipynb` file.

2.  **Run Cells:** Execute the cells in order to follow the project flow.

3.  **Experiment:** Feel free to modify the model architecture, hyperparameters (learning rate, number of epochs), and preprocessing steps to see how they impact the final accuracy.

## Saving and Loading the Model

The final step in the notebook demonstrates how to save the model's learned parameters to a `.pth` file, allowing you to load the trained model later without retraining.

```
# Save only the model parameters
torch.save(model.state_dict(), 'mnist_model_weights.pth')
```
