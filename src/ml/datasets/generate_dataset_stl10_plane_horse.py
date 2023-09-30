from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import torch
import os

import torchvision
import torchvision.transforms as transforms

from src.ml.datasets.generate_dataset import add_line_to_csv, create_latent_space_dataset, train_direction_matrix, \
    generate_dataset, train_and_save_gan
from src.ml.models.stl10.stl10_discriminator import Stl10Discriminator1
from src.ml.models.stl10.stl10_generator import Stl10Generator
from src.ml.models.stl10.stl10_reconstructor import Stl10Reconstructor

# Hyperparameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
learning_rate = 0.0005
gan_num_epochs = 5000
num_color_channels = 3
num_feature_maps_g = 32
num_feature_maps_d = 32
image_size = 96
size_z = 100
test_size = 1
directions_count = 50
direction_train_steps = 1500
num_imgs = 0

map_anomalies = True
map_normals = True
tmp_directory = '../data_backup'
data_root_directory = '../data'
dataset_name = 'DS7_stl10_plane_horse'


def generate_normals(dataset_folder, csv_path, temp_directory):
    stl10_dataset = torchvision.datasets.STL10(root=temp_directory, download=True, transform=transform)

    norm_class = 0
    norms = []
    for d in stl10_dataset:
        if d[1] == norm_class:
            norms.append(d)

    for i, img in enumerate(norms):
        img = (img[0] * 0.5) + 0.5
        img = transforms.ToPILImage()(img)
        file_name = f"img_{norm_class}_{i}.png"
        img.save(os.path.join(dataset_folder, file_name))
        add_line_to_csv(csv_path, [file_name, "False"])


def generate_anomalies(dataset_folder, csv_path, temp_directory, ano_fraction):
    stl10_dataset = torchvision.datasets.STL10(root=temp_directory, download=True, transform=transform)

    ano_class = 6
    anos = []
    for d in stl10_dataset:
        if d[1] == ano_class:
            anos.append(d)

    anos = anos[:round(len(anos)*ano_fraction)]
    for i, img in enumerate(anos):
        img = (img[0] * 0.5) + 0.5
        img = transforms.ToPILImage()(img)
        file_name = f"img_{ano_class}_{i}.png"
        img.save(os.path.join(dataset_folder, file_name))
        add_line_to_csv(csv_path, [file_name, "True"])


# ################## RUN ####################

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

stl10_generator = Stl10Generator(size_z=size_z, num_feature_maps=num_feature_maps_g).to(device)
stl10_discriminator = Stl10Discriminator1(num_feature_maps=num_feature_maps_d).to(device)
stl10_reconstructor = Stl10Reconstructor(directions_count=directions_count, width=2).to(device)


# generate_dataset(root_dir=data_root_directory,
#                  temp_directory=tmp_directory,
#                  dataset_name=dataset_name,
#                  generate_normals=generate_normals,
#                  generate_anomalies=generate_anomalies,
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
#                    generator=stl10_generator,
#                    discriminator=stl10_discriminator,
#                    transform=transform,
#                    num_imgs=num_imgs,
#                    save_checkpoint_every_n_epoch=50)

# train_direction_matrix(root_dir=data_root_directory,
#                        dataset_name=dataset_name,
#                        direction_count=directions_count,
#                        steps=direction_train_steps,
#                        device=device,
#                        use_bias=True,
#                        generator=stl10_generator,
#                        reconstructor=stl10_reconstructor)

# create_latent_space_dataset(root_dir=data_root_directory,
#                             transform=transform,
#                             dataset_name=dataset_name,
#                             size_z=size_z,
#                             num_feature_maps_g=num_feature_maps_g,
#                             num_feature_maps_d=num_feature_maps_d,
#                             num_color_channels=num_color_channels,
#                             device=device,
#                             max_opt_iterations=40000,
#                             generator=stl10_generator,
#                             discriminator=stl10_discriminator)

def test_generator(num, g, g_path):
    fixed_noise = torch.randn(num, size_z, 1, 1, device=device)
    g.load_state_dict(torch.load(g_path, map_location=torch.device(device)))
    fake_imgs = stl10_generator(fixed_noise).detach().cpu()
    with torch.no_grad():
        grid = torchvision.utils.make_grid(fake_imgs, nrow=8, normalize=True)
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)  # channel dim should be last
        plt.matshow(grid_np)
        plt.axis("off")
        plt.show()


test_generator(64, stl10_generator,
               '/home/yashar/git/AD-with-GANs/data/DS7_stl10_plane_horse/generator.pkl')
