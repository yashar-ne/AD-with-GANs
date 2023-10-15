import torchvision
import torch
import os

from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from src.ml.datasets.generate_dataset import add_line_to_csv, create_latent_space_dataset, train_direction_matrix, \
    generate_dataset, train_and_save_gan, test_generator

from src.ml.models.discriminator import Discriminator
from src.ml.models.generator import Generator
from src.ml.models.reconstructor import Reconstructor

# Hyperparameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
learning_rate = 0.0002
gan_num_epochs = 100
num_color_channels = 1
num_feature_maps_g = 64
num_feature_maps_d = 64
image_size = 28
size_z = 100
directions_count = 30
direction_train_steps = 2000
num_imgs = 0
max_opt_iterations = 10000

map_anomalies = True
map_normals = True
tmp_directory = '../data_backup'
data_root_directory = '../data'
dataset_name = 'DS8_fashion_mnist_shirt_sneaker'


def generate_normals(dataset_folder, csv_path, temp_directory):
    fashion_mnist_dataset = torchvision.datasets.FashionMNIST(root=temp_directory, download=True, transform=transform)

    norm_class = 0
    norms = []
    for d in fashion_mnist_dataset:
        if d[1] == norm_class:
            norms.append(d)

    for i, img in enumerate(norms):
        img = (img[0] * 0.5) + 0.5
        img = transforms.ToPILImage()(img)
        file_name = f"img_{norm_class}_{i}.png"
        img.save(os.path.join(dataset_folder, file_name))
        add_line_to_csv(csv_path, [file_name, "False"])


def generate_anomalies(dataset_folder, csv_path, temp_directory, ano_fraction):
    fashion_mnist_dataset = torchvision.datasets.FashionMNIST(root=temp_directory, download=True, transform=transform)

    ano_class = 7
    anos = []
    for d in fashion_mnist_dataset:
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

generator = Generator(size_z=size_z, num_feature_maps=num_feature_maps_g, num_color_channels=1).to(device)
discriminator = Discriminator(num_feature_maps=num_feature_maps_d, num_color_channels=1).to(device)
reconstructor = Reconstructor(directions_count=directions_count, width=2).to(device)

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
#                    generator=generator,
#                    discriminator=discriminator,
#                    transform=transform,
#                    num_imgs=num_imgs)


# train_direction_matrix(root_dir=data_root_directory,
#                        dataset_name=dataset_name,
#                        direction_count=directions_count,
#                        steps=direction_train_steps,
#                        device=device,
#                        use_bias=True,
#                        generator=generator,
#                        reconstructor=reconstructor)

create_latent_space_dataset(root_dir=data_root_directory,
                            transform=transform,
                            dataset_name=dataset_name,
                            size_z=size_z,
                            num_feature_maps_g=num_feature_maps_g,
                            num_feature_maps_d=num_feature_maps_d,
                            num_color_channels=num_color_channels,
                            device=device,
                            max_opt_iterations=max_opt_iterations,
                            generator=generator,
                            discriminator=discriminator,
                            num_images=num_imgs,
                            max_retries=3,
                            opt_threshold=0.06,
                            ignore_rules_below_threshold=0.08,
                            immediate_retry_threshold=0.1,
                            plateu_threshold=-1,
                            check_every_n_iter=5000,
                            learning_rate=0.001,
                            print_every_n_iters=5000,
                            retry_after_n_iters=5000,
                            draw_images=False
                            )

# test_generator(64, size_z, generator,
#                '/home/yashar/git/AD-with-GANs/data/DS8_fashion_mnist_shirt_sneaker/generator.pkl', device)
