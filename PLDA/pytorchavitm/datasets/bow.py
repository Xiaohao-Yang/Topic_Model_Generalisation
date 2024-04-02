import torch
from torch.utils.data import Dataset


class BOWDataset(Dataset):
    def __init__(self, X, idx2token):
        self.data = X
        self.idx2token = idx2token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        X = torch.FloatTensor(self.data[i])
        return {'X': X}