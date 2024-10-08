import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class AnoDataset(Dataset):
    def __init__(self, root_dir, transform=None, nrows=0, unpolluted=False):
        root_dir = os.path.join(root_dir)
        assert os.path.exists(os.path.join(root_dir, "ano_dataset.csv")), "Invalid root directory"
        self.root_dir = root_dir
        self.transform = transform

        df = pd.read_csv(os.path.join(root_dir, "ano_dataset.csv")).iloc[::-1]
        if unpolluted:
            df = df.loc[df['anomaly'] == False]
        self.label = df.head(nrows) if nrows > 0 else df

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.label.iloc[idx, 0])
        image_label = self.label.iloc[idx, 1]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, image_label, self.label.iloc[idx, 0]
