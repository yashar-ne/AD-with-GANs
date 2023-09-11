import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


class AnoMNIST(Dataset):
    def __init__(self, root_dir, transform=None, num_imgs=None):
        root_dir = os.path.join(root_dir)
        assert os.path.exists(os.path.join(root_dir, "anomnist_dataset.csv")), "Invalid root directory"
        self.root_dir = root_dir
        self.transform = transform
        self.label = pd.read_csv(os.path.join(root_dir, "anomnist_dataset.csv"), nrows=num_imgs)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.label.iloc[idx, 0])
        image_label = self.label.iloc[idx, 1]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, image_label
