import pandas as pd
import torch
from torch.utils.data import Dataset

class EmbeddDataset(Dataset):
    def __init__(self, pickle_file):

        self.data_df = pd.read_pickle(pickle_file)
        self.x_data = self.data_df['embedding'].values
        self.y_data = self.data_df['embedding'].values

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x_data[index])
        y = torch.FloatTensor(self.y_data[index])

        return x, y