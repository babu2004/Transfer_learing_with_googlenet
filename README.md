# Intel Image Classification using Transfer Learning (GoogLeNet)

This project implements **Transfer Learning using a pre-trained GoogLeNet model in PyTorch** to perform multi-class image classification on the Intel Image Classification dataset.

The goal is to classify images into six scenery categories using a fine-tuned deep convolutional neural network.

---

## Problem Statement

The objective of this project is to classify images into one of the following six categories:

- Mountain  
- Glacier  
- Street  
- Sea  
- Forest  
- Buildings  

This is a **multi-class image classification problem** where each image belongs to exactly one class.

---

## Dataset

The dataset used is the **Intel Image Classification dataset** from Kaggle.

It contains labeled images of natural and urban scenery divided into six categories listed above.

The dataset is split into:

- Training set  
- Validation set  
- Test set  

This ensures proper evaluation and avoids data leakage.

---

## Approach

### Transfer Learning

Instead of training a deep CNN from scratch, this project uses a **pre-trained GoogLeNet model** trained on ImageNet.

Why?
- Faster training  
- Better feature extraction  
- Improved performance with limited data  

---

### Model Modification

Since the original GoogLeNet model outputs 1000 ImageNet classes, the final fully connected layer was replaced to match the 6 target classes:

```python
googlenet_model.fc = torch.nn.Linear(googlenet_model.fc.in_features, num_class)
```

All layers were fine-tuned by enabling gradient updates for the entire network.

---

### Data Preparation

- Image paths and labels organized using a Pandas DataFrame  
- Labels encoded into numerical format  
- Train / validation / test split  
- Images resized to 128x128  
- Converted to PyTorch tensors  

---

### Training Details

- Loss Function: `CrossEntropyLoss`  
- Optimizer: `Adam`  
- Epochs: 15  
- Full fine-tuning of all layers  

Training and validation accuracy were monitored across epochs.

---

## Results

- Training Accuracy: ~94%  
- Validation Accuracy: ~86%  

The results demonstrate that transfer learning significantly improves performance while reducing training complexity compared to training from scratch.

---

## Prediction Pipeline

A prediction function was implemented to:

1. Load a new image  
2. Apply preprocessing transformations  
3. Pass it through the trained model  
4. Output the predicted class  

The trained model state dictionary was saved for future inference.

---

## Key Learnings

- Understanding how transfer learning accelerates deep learning workflows  
- Modifying pre-trained architectures for custom classification tasks  
- Difference between training from scratch and fine-tuning  
- Importance of proper dataset splitting  
- Real-world application of CNNs in computer vision  

---

## Future Improvements

- Experiment with freezing earlier layers instead of full fine-tuning  
- Try deeper architectures (ResNet, EfficientNet)  
- Add data augmentation for better generalization  
- Implement confusion matrix and per-class accuracy  
- Perform hyperparameter tuning  

---

## Conclusion

This project demonstrates how transfer learning can be effectively used to build powerful image classification systems without training deep networks from scratch.

It helped bridge the gap between theoretical deep learning concepts and practical, real-world computer vision applications.
