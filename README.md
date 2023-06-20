# Engine Condition Prediction using a Hybrid Transformer-CNN Model

This documentation provides an overview and implementation details of a hybrid model used for engine condition prediction. The model combines convolutional neural networks (CNNs) and transformers to achieve accurate predictions. The code snippets and explanations provided here can be used as a reference for implementing a similar model for engine condition prediction using the PyTorch framework.



## Purpose
The purpose of this code is to demonstrate the implementation of a hybrid Transformer-CNN model for predicting engine health based on input features. The model is trained and evaluated using a synthetic dataset of engine data.

## Dependencies
Before running the code, make sure you have the following dependencies installed:



```import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
```





## Code Explanation
The code is organized into the following sections:

**Importing Required Libraries**

**Custom Dataset Class**

**Creating an Instance of the Custom Dataset**

**Hyperparameters and Model Architecture**

**Splitting the Dataset**

**Creating Data Loaders**

**Checking for GPU Availability**

**Creating the Hybrid Model**

**Defining Loss Function and Optimizer**

**Training Loop**

**Testing**








## Results
The trained model will be evaluated on the test set, and the following metrics will be displayed:

**Loss**

**Accuracy**

**Precision**

**Recall**

**F1 Score**

**AUC**

**Confusion Matrix**





## Screenshots
Here are some screenshots of the code in action:

Training and Testing: This screenshot showcases the training and evaluation process, displaying the training and evaluation loss for each epoch. It also shows the testing results, including Loss, Accuracy, Precision, Recall, F1 Score, AUC, and Confusion matrix.

![image](https://github.com/DancesWithDobes/Pothole_binary_classification/assets/69741804/025a85ab-3fcc-4248-83ac-91dc90bf213d)



## Conclusion
This Colab file provides a comprehensive implementation of a hybrid model for engine condition prediction. 









