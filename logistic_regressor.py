import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Sample data (replace with your actual data loading)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]], dtype=np.float32)
y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train).unsqueeze(1)
y_test = torch.from_numpy(y_test).unsqueeze(1)

# Create Dataset and DataLoader
class BinaryClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples

train_dataset = BinaryClassificationDataset(X_train, y_train)
test_dataset = BinaryClassificationDataset(X_test, y_test)

batch_size = 2
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shu