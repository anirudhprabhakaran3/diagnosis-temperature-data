import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .constants import BATCH_SIZE

class TemperatureDataset(Dataset):
    def __init__(self, X_data, y_data):
        super(TemperatureDataset, self).__init__()
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    
    def __len__(self):
        return len(self.X_data)

def get_dataset(x, y):
    return TemperatureDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())

def get_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def scaffold_dataloaders(x_train_path, x_test_path, y_train_path, y_test_path):
    x_train = pd.read_csv(x_train_path, index_col=False).to_numpy()
    x_test = pd.read_csv(x_test_path, index_col=False).to_numpy()
    y_train = pd.read_csv(y_train_path, index_col=False).to_numpy()
    y_test = pd.read_csv(y_test_path, index_col=False).to_numpy()

    train_dataset = get_dataset(x_train, y_train)
    test_dataset = get_dataset(x_test, y_test)

    train_loader = get_dataloader(train_dataset)
    test_lodaer = get_dataloader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_lodaer
