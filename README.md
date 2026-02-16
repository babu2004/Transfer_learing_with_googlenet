```markdown
# Intel Image Classification using GoogLeNet (Transfer Learning)

This notebook demonstrates an image classification task using transfer learning with a pre-trained GoogLeNet model in PyTorch.

## Project Description

The goal of this project is to classify images into six distinct categories: `mountain`, `glacier`, `street`, `sea`, `forest`, and `buildings`. This is achieved by leveraging the power of transfer learning, specifically by fine-tuning a pre-trained GoogLeNet model on a custom dataset.

## Dataset

The dataset used for this project is sourced from Kaggle: `Intel Image Classification` (specifically, labeled scenery images). It includes images categorized into the six classes mentioned above.

## Key Concepts Implemented

-   **Transfer Learning**: Utilizing a pre-trained convolutional neural network (GoogLeNet) as a feature extractor and adapting it for a new classification task.
-   **Model Fine-tuning**: Modifying the final classification layer of the pre-trained model to match the number of target classes in our dataset and training the entire network.
-   **PyTorch Framework**: Implementation of the model, data loading, training loop, and evaluation using PyTorch.
-   **Data Preprocessing**: Image transformations (resizing, converting to tensor) for compatibility with the neural network.
-   **Custom Dataset and DataLoader**: Creating a custom `Dataset` class to handle image loading and labeling, and using `DataLoader` for efficient batch processing during training.

## Implementation Details

1.  **Data Loading and Preparation**: Images are loaded from the specified directory structure, and labels are extracted. A Pandas DataFrame is created to manage image paths and their corresponding labels.
2.  **Train-Validation-Test Split**: The dataset is split into training, validation, and test sets to ensure robust model evaluation.
3.  **Label Encoding**: Categorical labels are converted into numerical format using `LabelEncoder`.
4.  **Image Transformations**: Images are resized to 128x128 pixels and converted to PyTorch tensors.
5.  **GoogLeNet Model**: A pre-trained GoogLeNet model from `torchvision.models` is loaded.
6.  **Output Layer Modification**: The final fully connected layer of GoogLeNet is replaced to output 6 classes instead of the original 1000 ImageNet classes.
7.  **Training**: The model is trained using `CrossEntropyLoss` and the `Adam` optimizer for 15 epochs. Training loss and accuracy are monitored.
8.  **Evaluation**: The model's performance is evaluated on the validation set after training.
9.  **Model Saving**: The trained model's state dictionary is saved.
10. **Prediction Function**: A utility function `predict_image` is provided to load an image, preprocess it, and make a prediction using the trained model, along with visualizing the image.

## Results

The model achieved a training accuracy of approximately 94.1% and a validation accuracy of around 86.46%.

## Usage

To run this notebook:

1.  Ensure you have access to the dataset specified or adapt the data loading paths to your own dataset.
2.  Install required libraries (e.g., `torch`, `torchvision`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `Pillow`).
3.  Execute the cells sequentially.

## Dependencies

-   `torch`
-   `torchvision`
-   `pandas`
-   `numpy`
-   `matplotlib`
-   `sklearn`
-   `Pillow`
