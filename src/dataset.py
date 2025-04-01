import pandas as pd
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        return text, label
