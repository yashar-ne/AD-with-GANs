from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import torch
import os

import torchvision
import torchvision.transforms as transforms

from src.ml.datasets.generate_dataset import add_line_to_csv, create_latent_space_dataset, train_direction_matrix, \
    generate_dataset, train_and_save_gan, test_generator
from src.ml.models.mvtec128.mvtec_discriminator import MvTecDiscriminator
from src.ml.models.mvtec128.mvtec_generator import MvTecGenerator
from src.ml.models.mvtec128.mvtec_reconstructor import MvTecReconstructor

# Hyperparameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 256
learning_rate = 0.0005
gan_num_epochs = 2000
num_color_channels = 3
num_feature_maps_g = 64
num_feature_maps_d = 64
image_size = 128
size_z = 100
directions_count = 10
direction_train_steps = 2500
num_imgs = 0

map_anomalies = True
map_normals = True
tmp_directory = '../data_backup'
data_root_directory = '../data'
dataset_name = 'DS9_mvtec_hazelnut'


def generate_normals(dataset_folder, csv_path, temp_directory):
    mvtec_hazelnut_normals_folder = os.path.join(temp_directory, "mvtec_hazelnut", "normals")
    for counter, filename in enumerate(os.listdir(mvtec_hazelnut_normals_folder)):
        if filename.endswith(".png"):
            file_name = f"img_norm_{counter}_0.png"
            img = Image.open(os.path.join(mvtec_hazelnut_normals_folder, filename))
            img.thumbnail((128, 128), Image.Resampling.LANCZOS)
            img.save(os.path.join(dataset_folder, file_name))
            add_line_to_csv(csv_path, [file_name, "False"])

            # file_name = f"img_norm_{counter}_0_1.png"
            # img_mirror = ImageOps.mirror(img)
            # img_mirror.save(os.path.join(dataset_folder, file_name))
            # add_line_to_csv(csv_path, [file_name, "False"])
            #
            # file_name = f"img_norm_{counter}_1.png"
            # img = img.rotate(90)
            # img.save(os.path.join(dataset_folder, file_name))
            # add_line_to_csv(csv_path, [file_name, "False"])
            #
            # file_name = f"img_norm_{counter}_1_1.png"
            # img_mirror = ImageOps.mirror(img)
            # img_mirror.save(os.path.join(dataset_folder, file_name))
            # add_line_to_csv(csv_path, [file_name, "False"])
            #
            # file_name = f"img_norm_{counter}_2.png"
            # img = img.rotate(90)
            # img.save(os.path.join(dataset_folder, file_name))
            # add_line_to_csv(csv_path, [file_name, "False"])
            #
            # file_name = f"img_norm_{counter}_2_1.png"
            # img_mirror = ImageOps.mirror(img)
            # img_mirror.save(os.path.join(dataset_folder, file_name))
            # add_line_to_csv(csv_path, [file_name, "False"])
            #
            # file_name = f"img_norm_{counter}_3.png"
            # img = img.rotate(90)
            # img.save(os.path.join(dataset_folder, file_name))
            # add_line_to_csv(csv_path, [file_name, "False"])
            #
            # file_name = f"img_norm_{counter}_3_1.png"
            # img_mirror = ImageOps.mirror(img)
            # img_mirror.save(os.path.join(dataset_folder, file_name))
            # add_line_to_csv(csv_path, [file_name, "False"])


def generate_anomalies(dataset_folder, csv_path, temp_directory, ano_fraction):
    mvtec_hazelnut_anomalies_folder = os.path.join(temp_directory, "mvtec_hazelnut", "anomalies")
    counter = 0
    for _, dirs, files in os.walk(mvtec_hazelnut_anomalies_folder):
        for directory in dirs:
            sub_folder = os.path.join(mvtec_hazelnut_anomalies_folder, directory)
            for i, filename in enumerate(os.listdir(sub_folder)):
                if filename.endswith(".png"):
                    file_name = f"img_ano_{counter}_0.png"
                    img = Image.open(os.path.join(sub_folder, filename))
                    img.thumbnail((128, 128), Image.Resampling.LANCZOS)
                    img.save(os.path.join(dataset_folder, file_name))
                    add_line_to_csv(csv_path, [file_name, "True"])

                    file_name = f"img_ano_{counter}_0_1.png"
                    img_mirror = ImageOps.mirror(img)
                    img_mirror.save(os.path.join(dataset_folder, file_name))
                    add_line_to_csv(csv_path, [file_name, "True"])

                    file_name = f"img_ano_{counter}_1.png"
                    img = img.rotate(90)
                    img.save(os.path.join(dataset_folder, file_name))
                    add_line_to_csv(csv_path, [file_name, "True"])

                    file_name = f"img_ano_{counter}_1_1.png"
                    img_mirror = ImageOps.mirror(img)
                    img_mirror.save(os.path.join(dataset_folder, file_name))
                    add_line_to_csv(csv_path, [file_name, "True"])

                    file_name = f"img_ano_{counter}_2.png"
                    img = img.rotate(90)
                    img.save(os.path.join(dataset_folder, file_name))
                    add_line_to_csv(csv_path, [file_name, "True"])

                    file_name = f"img_ano_{counter}_2_1.png"
                    img_mirror = ImageOps.mirror(img)
                    img_mirror.save(os.path.join(dataset_folder, file_name))
                    add_line_to_csv(csv_path, [file_name, "True"])

                    file_name = f"img_ano_{counter}_3.png"
                    img = img.rotate(90)
                    img.save(os.path.join(dataset_folder, file_name))
                    add_line_to_csv(csv_path, [file_name, "True"])

                    file_name = f"img_ano_{counter}_3_1.png"
                    img_mirror = ImageOps.mirror(img)
                    img_mirror.save(os.path.join(dataset_folder, file_name))
                    add_line_to_csv(csv_path, [file_name, "True"])

                    counter += 1


# ################## RUN ####################

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

mvtec_128_generator = MvTecGenerator(size_z=size_z, num_feature_maps=num_feature_maps_g).to(device)
mvtec_128_discriminator = MvTecDiscriminator(num_feature_maps=num_feature_maps_d).to(device)
mvtec_128_reconstructor = MvTecReconstructor(directions_count=directions_count, width=2).to(device)


# generate_dataset(root_dir=data_root_directory,
#                  temp_directory=tmp_directory,
#                  dataset_name=dataset_name,
#                  generate_normals=generate_normals,
#                  generate_anomalies=generate_anomalies,
#                  ano_fraction=0.1)
#
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
#                    generator=mvtec_128_generator,
#                    discriminator=mvtec_128_discriminator,
#                    transform=transform,
#                    num_imgs=num_imgs,
#                    save_checkpoint_every_n_epoch=50)

train_direction_matrix(root_dir=data_root_directory,
                       dataset_name=dataset_name,
                       direction_count=directions_count,
                       steps=direction_train_steps,
                       device=device,
                       use_bias=True,
                       generator=mvtec_128_generator,
                       reconstructor=mvtec_128_reconstructor)

# create_latent_space_dataset(root_dir=data_root_directory,
#                             transform=transform,
#                             dataset_name=dataset_name,
#                             size_z=size_z,
#                             num_feature_maps_g=num_feature_maps_g,
#                             num_feature_maps_d=num_feature_maps_d,
#                             num_color_channels=num_color_channels,
#                             device=device,
#                             max_opt_iterations=10000,
#                             generator=mvtec_128_generator,
#                             discriminator=mvtec_128_discriminator)

# test_generator(128, size_z, mvtec_128_generator,
#                '/home/yashar/git/AD-with-GANs/data/DS9_mvtec_hazelnut/generator.pkl', device)
