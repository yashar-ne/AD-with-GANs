import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class LatentSpaceMNIST(Dataset):
    def __init__(self, root_dir, transform=None):
        root_dir = os.path.join(root_dir, "LatentSpaceMNIST")
        assert os.path.exists(os.path.join(root_dir, "latent_space_mappings.csv")), "Invalid root directory"
        self.root_dir = root_dir
        self.transform = transform
        self.label = pd.read_csv(os.path.join(root_dir, "latent_space_mappings.csv"))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        latent_space_point_path = os.path.join(self.root_dir, self.label.iloc[idx, 0])
        latent_space_point = np.reshape(torch.load(latent_space_point_path, map_location=torch.device(self.device)).detach().numpy(), 100)
        latent_space_point_label = {"label": self.label.iloc[idx, 1], "anomaly": self.label.iloc[idx, 2]}

        if self.transform:
            latent_space_point = self.transform(latent_space_point)

        return latent_space_point, latent_space_point_label
