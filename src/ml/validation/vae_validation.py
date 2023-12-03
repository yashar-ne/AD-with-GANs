import csv
import os

import torch
import torchvision
from PIL import Image

from src.ml.models.base.beta_vae64 import BetaVAE64
from src.ml.tools.utils import get_folders_from_dataset_name
from src.ml.validation.validation_utils import get_roc_curve_as_base64


def get_vae_roc_auc_for_image_data(root_dir, dataset_name, vae=None):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    dataset_folder, dataset_raw_folder, checkpoint_folder = get_folders_from_dataset_name(root_dir, dataset_name)

    if vae is None:
        vae: BetaVAE64 = torch.load(os.path.join(dataset_folder, 'vae_model.pkl'))

    vae.eval()
    vae.cpu()
    image_scores = []
    y = []
    # load dataset csv and iterate it
    csv_file_path = os.path.join(dataset_raw_folder, 'ano_dataset.csv')
    with open(csv_file_path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)
        for row in datareader:
            image_path = os.path.join(dataset_raw_folder, row[0])
            img = Image.open(image_path)
            image_scores.append(vae.get_reconstruction_loss(transform(img)).detach().numpy())
            y.append(1 if row[1] == 'True' else 0)

    return get_roc_curve_as_base64(y, image_scores)
