import csv
import os
import random
import shutil

import torch
from src.ml.dataset_generation.ano_mnist import AnoDataset
from torch.utils.data import RandomSampler
from torchvision import datasets
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def generate_lined_mnist_images(base_folder, num, max_augmentation_thickness=5,
                                randomize_augmentation_thickness=False, labels=[]):
    assert max_augmentation_thickness <= 7, "max_augmentation_thickness must be smaller than 7"
    os.makedirs(base_folder, exist_ok=True)

    dataset = datasets.MNIST(
        root=base_folder,
        train=True,
        download=True,
    )

    if len(labels) > 0:
        dataset = [d for d in dataset if (d[1] in labels)]
    else:
        dataset = dataset.data

    ano_mnist_drop_folder = os.path.join(base_folder, "AnoMNIST")
    csv_path = os.path.join(ano_mnist_drop_folder, "ano_dataset.csv")

    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(ano_mnist_drop_folder, exist_ok=True)

    augmentation_thickness: int = random.randint(1, max_augmentation_thickness)
    for i in range(num):
        random_idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[random_idx]

        augmentation_thickness = random.randint(3,
                                                max_augmentation_thickness) if randomize_augmentation_thickness else augmentation_thickness
        random_idx = random.randint(4, 20)
        for j in range(img.size[0]):
            for k in range(augmentation_thickness):
                img.putpixel((j, random_idx + k + 1), 0)

        img.save(os.path.join(ano_mnist_drop_folder, f"img_aug_{label}_{i}.png"))
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            fields = [f'img_aug_{label}_{i}.png', f"{label}", "True"]
            writer.writerow(fields)


def generate_lined_image_files(base_folder, num, labels=[]):
    ano_mnist_drop_folder = os.path.join(base_folder, "AnoMNIST")
    csv_path = os.path.join(ano_mnist_drop_folder, "ano_dataset.csv")

    if os.path.exists(ano_mnist_drop_folder):
        shutil.rmtree(ano_mnist_drop_folder)

    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(ano_mnist_drop_folder, exist_ok=True)

    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        fields = ["filename", "label", "anomaly"]
        writer.writerow(fields)

    generate_lined_mnist_images(base_folder, num=num, labels=labels)


def get_ano_mnist_dataset(transform, root_dir, labels=[9], train_size=0.9):
    ano_mnist_dataset = AnoDataset(
        root_dir=root_dir,
        transform=transform
    )

    mnist_dataset = AnomalyExtendedMNIST(
        root=root_dir,
        train=True,
        transform=transform,
        download=True,
    )

    dat = torch.utils.data.ConcatDataset([ano_mnist_dataset, mnist_dataset])

    if len(labels) > 0:
        dat = [d for d in dat if (d[1]['label'] in labels)]

    absolute_train_size = int(len(dat) * train_size)
    absolute_test_size = len(dat) - absolute_train_size
    return torch.utils.data.random_split(dat, [absolute_train_size, absolute_test_size])


def get_ano_class_mnist_dataset(root_dir, transform, norm_class=9, ano_class=6, ano_fraction=0.1):
    mnist_dataset = MNIST(
        root=root_dir,
        train=True,
        transform=transform,
        download=True,
    )

    norms = [d for d in mnist_dataset if (d[1] == norm_class)]
    anos = [d for d in mnist_dataset if (d[1] == ano_class)]

    return torch.utils.data.ConcatDataset([norms, anos[:round(ano_fraction * len(anos))]])


# generate_anomalous_image_files(base_folder='/home/yashar/git/python/AD-with-GANs/data', num=2000, labels=[9]) # num normals 5949
get_ano_class_mnist_dataset(
    root_dir='/home/yashar/git/python/AD-with-GANs/data',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(.5,), std=(.5,))
    ])
)
