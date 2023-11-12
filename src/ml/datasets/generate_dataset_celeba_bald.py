import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.ml.datasets.generate_dataset import add_line_to_csv, train_direction_matrix
from src.ml.models.celebA.celeb_discriminator import CelebDiscriminator
from src.ml.models.celebA.celeb_generator import CelebGenerator
from src.ml.models.celebA.celeb_reconstructor import CelebReconstructor

# Hyperparameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
learning_rate = 0.0002
gan_num_epochs = 100
num_color_channels = 3
num_feature_maps_g = 64
num_feature_maps_d = 64
image_size = 64
size_z = 100
test_size = 1
directions_count = 500
direction_train_steps = 15000
num_imgs = 101300

map_anomalies = True
map_normals = True
tmp_directory = '../data_temp'
data_root_directory = '../data'
dataset_name = 'DS5_celebA_bald'


class AnoCelebA(Dataset):
    def __init__(self, root_dir, transform=None, nrows=None):
        root_dir = os.path.join(root_dir)
        assert os.path.exists(os.path.join(root_dir, "celebA")), "Invalid root directory"
        self.root_dir = root_dir
        self.transform = transform
        self.label = pd.read_csv(os.path.join(root_dir, "celebA/list_attr_celeba.csv"), nrows=nrows)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'celebA/imgs', self.label.iloc[idx, 0])
        image_label = self.label.iloc[idx, 5]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, image_label


def generate_normals(dataset_folder, csv_path, temp_directory):
    celeba_dataset = AnoCelebA(
        root_dir=temp_directory,
    )

    norm_class = -1
    norms = []
    for d in celeba_dataset:
        if d[1] == norm_class:
            norms.append(d)

    for i, img in enumerate(norms):
        file_name = f"img_{norm_class}_{i}.png"
        img[0].save(os.path.join(dataset_folder, file_name))
        add_line_to_csv(csv_path, [file_name, "False"])


def generate_anomalies_bald(dataset_folder, csv_path, temp_directory, ano_fraction):
    celeba_dataset = AnoCelebA(
        root_dir=temp_directory,
    )

    ano_class = 1
    anos = []
    for d in celeba_dataset:
        if d[1] == ano_class:
            anos.append(d)

    for i, img in enumerate(anos):
        file_name = f"img_{ano_class}_{i}.png"
        img[0].save(os.path.join(dataset_folder, file_name))
        add_line_to_csv(csv_path, [file_name, "True"])


# ################## RUN ####################

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

celeb_generator = CelebGenerator(size_z=size_z, num_feature_maps=num_feature_maps_g).to(device)
celeb_discriminator = CelebDiscriminator(num_feature_maps=num_feature_maps_d).to(device)
celeb_reconstructor = CelebReconstructor(directions_count=directions_count, width=2).to(device)

# generate_dataset(root_dir=data_root_directory,
#                  temp_directory=tmp_directory,
#                  dataset_name=dataset_name,
#                  generate_normals=generate_normals,
#                  generate_anomalies=generate_anomalies_bald,
#                  ano_fraction=0.1)

# train_and_save_gan(root_dir=data_root_directory,
#                    dataset_name=dataset_name,
#                    size_z=size_z,
#                    num_epochs=gan_num_epochs,
#                    num_feature_maps_g=num_feature_maps_g,
#                    num_feature_maps_d=num_feature_maps_d,
#                    num_color_channels=num_color_channels,
#                    batch_size=batch_size,
#                    device=device,
#                    learning_rate=learning_rate,
#                    generator=celeb_generator,
#                    discriminator=celeb_discriminator,
#                    transform=transform,
#                    num_imgs=num_imgs)


train_direction_matrix(root_dir=data_root_directory,
                       dataset_name=dataset_name,
                       direction_count=directions_count,
                       steps=direction_train_steps,
                       device=device,
                       use_bias=True,
                       generator=celeb_generator,
                       reconstructor=celeb_reconstructor)

# create_latent_space_dataset(root_dir=data_root_directory,
#                             transform=transform,
#                             dataset_name=dataset_name,
#                             size_z=size_z,
#                             num_feature_maps_g=num_feature_maps_g,
#                             num_feature_maps_d=num_feature_maps_d,
#                             num_color_channels=num_color_channels,
#                             device=device,
#                             max_opt_iterations=20000,
#                             generator=celeb_generator,
#                             discriminator=celeb_discriminator,
#                             num_images=num_imgs,
#                             start_with_image_number=782)
