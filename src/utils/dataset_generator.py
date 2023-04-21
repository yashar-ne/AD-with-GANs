from torchvision import datasets
from PIL import Image, ImageDraw

import os
import shutil
import random


def generate_augmented_mnist_images(drop_folder, num=10, max_augmentation_thickness=5,
                                    randomize_augmentation_thickness=False):
    assert max_augmentation_thickness <= 7, "max_augmentation_thickness must be smaller than 7"
    dataset = datasets.MNIST(
        root='../data',
        train=True,
        download=False,
    )

    shutil.rmtree(drop_folder)
    if not os.path.exists(drop_folder):
        os.makedirs(drop_folder)

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

        img.save(f"{drop_folder}img_aug_{label}_{i}.png")


def generate_artificial_mnist_images(drop_folder, num=10):
    for i in range(num):
        label = random.randint(0, 9)
        img = Image.new("1", (28, 28))
        d = ImageDraw.Draw(img)
        d.text((5, 5), f'{label}', fill=1)

        img.save(f"{drop_folder}img_art_{label}_{i}.png")


def generate_dataset(drop_folder='../data/MNIST_AUGMENTED/', num_aug=10, num_art=10):
    # generate_augmented_mnist_images(drop_folder, num=num_aug)
    generate_artificial_mnist_images(drop_folder, num=num_art)


generate_dataset()
