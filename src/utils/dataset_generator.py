from PIL import Image, ImageDraw, ImageFont
from AnoMNIST import AnoMNIST
from torchvision import datasets, transforms
import torch

import os
import shutil
import random
import csv


def generate_augmented_mnist_images(drop_folder, num=10, max_augmentation_thickness=5,
                                    randomize_augmentation_thickness=False):
    assert max_augmentation_thickness <= 7, "max_augmentation_thickness must be smaller than 7"
    dataset = datasets.MNIST(
        root='../data',
        train=True,
        download=False,
    )

    augmentation_thickness: int = random.randint(1, max_augmentation_thickness)
    for i in range(num):
        random_idx = random.randint(0, len(dataset.data) - 1)
        img, label = dataset[random_idx]

        augmentation_thickness = random.randint(1,
                                                max_augmentation_thickness) if randomize_augmentation_thickness else augmentation_thickness
        random_idx = random.randint(4, 20)
        for j in range(img.size[0]):
            for k in range(augmentation_thickness):
                img.putpixel((j, random_idx + k + 1), 0)

        img.save(f"{drop_folder}/img_aug_{label}_{i}.png")
        with open(f'{drop_folder}anomnist_dataset.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            fields = [f'img_aug_{label}_{i}.png', f"{label}", "True", "Augmented"]
            writer.writerow(fields)


def generate_artificial_mnist_images(drop_folder, num=10):
    for i in range(num):
        label = random.randint(0, 9)
        img = Image.new("1", (28, 28))

        font = ImageFont.truetype("./FFFFORWA.TTF", size=18)
        d = ImageDraw.Draw(img)
        d.text((8, 4), f'{label}', fill=1, font=font)

        img.save(f"{drop_folder}/img_art_{label}_{i}.png")
        with open(f'{drop_folder}anomnist_dataset.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            fields = [f'img_art_{label}_{i}.png', f"{label}", "True", "Artificial"]
            writer.writerow(fields)


def generate_anomalous_image_files(drop_folder='../data/AnoMNIST/', num_aug=100, num_art=100):
    if os.path.exists(drop_folder):
        shutil.rmtree(drop_folder)
    if not os.path.exists(f"{drop_folder}"):
        os.makedirs(f"{drop_folder}")

    with open(f'{drop_folder}anomnist_dataset.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        fields = ["filename", "label", "anomaly", "type"]
        writer.writerow(fields)

    generate_augmented_mnist_images(drop_folder, num=num_aug)
    generate_artificial_mnist_images(drop_folder, num=num_art)


def get_ano_mnist_dataset(transform, root_dir):
    ano_mnist_dataset = AnoMNIST(
        root_dir=root_dir,
        transform=transform
    )

    mnist_dataset = datasets.MNIST(
        root=root_dir,
        train=True,
        transform=transform,
        download=False,
    )

    return torch.utils.data.ConcatDataset([ano_mnist_dataset, mnist_dataset])
