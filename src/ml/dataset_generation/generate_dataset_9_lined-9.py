import os
import random

import torch
from torchvision.datasets import MNIST

from src.ml.dataset_generation.generate_dataset import add_line_to_csv, train_direction_matrix

# Hyperparameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 512
num_classes = 1
learning_rate = 0.0001
num_epochs = 100
num_color_channels = 1
num_feature_maps_g = 64
num_feature_maps_d = 64
size_z = 100
test_size = 1

map_anomalies = True
map_normals = True
temp_directory = '../../../data_temp'
data_root_directory = '../../../data'
dataset_name = 'DS4_mnist_9_9lined_10_percent'


def generate_normals_9(dataset_folder, csv_path, temp_directory):
    mnist_dataset = MNIST(
        root=temp_directory,
        train=True,
        download=True,
    )

    norm_class = 9
    norms = [d for d in mnist_dataset if (d[1] == norm_class)]
    for i, img in enumerate(norms):
        file_name = f"img_{norm_class}_{i}.png"
        img[0].save(os.path.join(dataset_folder, file_name))
        add_line_to_csv(csv_path, [file_name, "False"])


def generate_anomalies_6(dataset_folder, csv_path, temp_directory, ano_fraction):
    mnist_dataset = MNIST(
        root=temp_directory,
        train=True,
        download=True,
    )

    ano_class = 9
    max_augmentation_thickness = 8
    randomize_augmentation_thickness = False
    augmentation_thickness: int = random.randint(1, max_augmentation_thickness)
    anos = [d for d in mnist_dataset if (d[1] == ano_class)]
    anos = anos[:round(len(anos) * ano_fraction)]
    for i, (img, label) in enumerate(anos):
        augmentation_thickness = random.randint(3,
                                                max_augmentation_thickness) if randomize_augmentation_thickness else augmentation_thickness
        random_idx = random.randint(4, 20)
        for j in range(img.size[0]):
            for k in range(augmentation_thickness):
                img.putpixel((j, random_idx + k + 1), 0)

        file_name = f"img_{ano_class}-lined_{i}.png"
        img.save(os.path.join(dataset_folder, file_name))
        add_line_to_csv(csv_path, [file_name, "True"])


# generate_dataset(root_dir=data_root_directory,
#                  temp_directory=temp_directory,
#                  dataset_name=dataset_name,
#                  generate_normals=generate_normals_9,
#                  generate_anomalies=generate_anomalies_6,
#                  ano_fraction=0.1)
#
# train_and_save_gan(root_dir=data_root_directory,
#                    dataset_name=dataset_name,
#                    size_z=size_z,
#                    num_epochs=150,
#                    num_feature_maps_g=64,
#                    num_feature_maps_d=64,
#                    num_color_channels=1,
#                    batch_size=batch_size,
#                    device=device,
#                    learning_rate=0.001)

train_direction_matrix(root_dir=data_root_directory,
                       dataset_name=dataset_name,
                       direction_count=50,
                       steps=3000,
                       device=device,
                       use_bias=True)

# create_latent_space_dataset(root_dir=data_root_directory,
#                             dataset_name=dataset_name,
#                             batch_size=batch_size,
#                             size_z=size_z,
#                             num_feature_maps_g=num_feature_maps_g,
#                             num_feature_maps_d=num_feature_maps_d,
#                             num_color_channels=num_color_channels,
#                             device=device)
