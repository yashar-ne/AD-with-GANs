import os

import torch
import torchvision
import torchvision.transforms as transforms

from src.ml.dataset_generation.dataset_generation_core import add_line_to_csv, test_generator_and_show_plot
from src.ml.models.cifar10.cifar10_discriminator import Cifar10Discriminator
from src.ml.models.cifar10.cifar10_generator import Cifar10Generator
from src.ml.models.cifar10.cifar10_reconstructor import Cifar10Reconstructor

# Hyperparameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 2048
learning_rate = 0.0002
gan_num_epochs = 1000
num_color_channels = 3
num_feature_maps_g = 64
num_feature_maps_d = 64
image_size = 32
size_z = 100
test_size = 1
directions_count = 500
direction_train_steps = 15000
num_imgs = 10000

map_anomalies = True
map_normals = True
tmp_directory = '../data_temp'
data_root_directory = '../data'
dataset_name = 'DS6_cifar10_plane_horse'


def generate_normals(dataset_folder, csv_path, temp_directory):
    cifar10_dataset = torchvision.datasets.CIFAR10(root=temp_directory, train=True,
                                                   download=True, transform=transform)

    norm_class = 1
    norms = []
    for d in cifar10_dataset:
        if d[1] == norm_class:
            norms.append(d)

    for i, img in enumerate(norms):
        img = (img[0] * 0.5) + 0.5
        img = transforms.ToPILImage()(img)
        file_name = f"img_{norm_class}_{i}.png"
        img.save(os.path.join(dataset_folder, file_name))
        add_line_to_csv(csv_path, [file_name, "False"])


def generate_anomalies(dataset_folder, csv_path, temp_directory, ano_fraction):
    cifar10_dataset = torchvision.datasets.CIFAR10(root=temp_directory, train=True,
                                                   download=True, transform=transform)

    ano_class = 6
    anos = []
    for d in cifar10_dataset:
        if d[1] == ano_class:
            anos.append(d)

    anos = anos[:round(len(anos) * ano_fraction)]
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

cifar10_generator = Cifar10Generator(size_z=size_z, num_feature_maps=num_feature_maps_g).to(device)
cifar10_discriminator = Cifar10Discriminator(num_feature_maps=num_feature_maps_d).to(device)
cifar10_reconstructor = Cifar10Reconstructor(directions_count=directions_count, width=2).to(device)

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
#                    generator=cifar10_generator,
#                    discriminator=cifar10_discriminator,
#                    transform=transform,
#                    num_imgs=num_imgs,
#                    save_checkpoint_every_n_epoch=50)

# train_direction_matrix(root_dir=data_root_directory,
#                        dataset_name=dataset_name,
#                        direction_count=directions_count,
#                        steps=direction_train_steps,
#                        device=device,
#                        use_bias=True,
#                        generator=celeb_generator,
#                        reconstructor=celeb_reconstructor)

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

test_generator_and_show_plot(64, size_z, cifar10_generator,
                             '/home/yashar/git/AD-with-GANs/data/DS6_cifar10_plane_horse/generator.pkl', device)
