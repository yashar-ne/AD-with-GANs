import numpy as np
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torchvision.io import read_image
import torchvision.utils as vutils
from torch.utils.data import Dataset
import pandas as pd
import torch
import os

from torchvision.transforms import transforms

from src.ml.datasets.generate_dataset import train_and_save_gan, add_line_to_csv, generate_dataset, \
    train_direction_matrix, create_latent_space_dataset

# Hyperparameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
learning_rate = 0.0002
gan_num_epochs = 10
num_color_channels = 3
num_feature_maps_g = 64
num_feature_maps_d = 64
image_size = 64
size_z = 100
test_size = 1
directions_count = 110
num_imgs = None  # None = Take all images

map_anomalies = True
map_normals = True
tmp_directory = '../../../data_backup'
data_root_directory = '../../../data'
dataset_name = 'DS5_celebA_bald'

celebA_directory = os.path.join(tmp_directory, 'celebA')
celebA_imgs_directory = os.path.join(celebA_directory, 'imgs')


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


class CelebGenerator(nn.Module):
    def __init__(self):
        super(CelebGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(size_z, num_feature_maps_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_feature_maps_g * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(num_feature_maps_g * 8, num_feature_maps_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps_g * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(num_feature_maps_g * 4, num_feature_maps_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps_g * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(num_feature_maps_g * 2, num_feature_maps_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps_g),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(num_feature_maps_g, num_color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

    def gen_shifted(self, x, shift):
        shift = torch.unsqueeze(shift, -1)
        shift = torch.unsqueeze(shift, -1)
        return self.forward(x + shift)


class CelebDiscriminator(nn.Module):
    def __init__(self):
        super(CelebDiscriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_color_channels, num_feature_maps_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_feature_maps_d, num_feature_maps_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_feature_maps_d * 2, num_feature_maps_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(num_feature_maps_d * 4, num_feature_maps_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Conv2d(num_feature_maps_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        feature = out
        out = self.fc(out)
        # return out.view(-1, 1).squeeze(1), feature
        return out, feature


class CelebReconstructor(nn.Module):
    def __init__(self, channels=num_color_channels, width=2):
        super(CelebReconstructor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, 3 * width, kernel_size=2),
            nn.BatchNorm2d(3 * width),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3 * width, 8 * width, kernel_size=2),
            nn.BatchNorm2d(8 * width),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8 * width, 60 * width, kernel_size=4),
            nn.BatchNorm2d(60 * width),
            nn.ReLU()
        )

        self.fc_logits = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            # nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, np.prod(directions_count))
        )
        self.fc_shift = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            # nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, 1)
        )

    def forward(self, x1, x2):
        bs = x1.shape[0]
        combined = torch.cat([x1, x2], dim=1)
        features = self.conv(combined)
        features = features.mean(dim=[-1, -2])
        features = features.view(bs, -1)

        logits = self.fc_logits(features)
        shift = self.fc_shift(features)

        return logits, shift.squeeze()


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


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

celeb_generator = CelebGenerator().to(device)
celeb_discriminator = CelebDiscriminator().to(device)
celeb_reconstructor = CelebReconstructor(width=2).to(device)


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


# train_direction_matrix(root_dir=data_root_directory,
#                        dataset_name=dataset_name,
#                        direction_count=directions_count,
#                        steps=10000,
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
#                             max_opt_iterations=100000,
#                             generator=celeb_generator,
#                             discriminator=celeb_discriminator)

def test_generator(num, g, g_path):
    fixed_noise = torch.randn(num, size_z, 1, 1, device=device)
    g.load_state_dict(torch.load(g_path, map_location=torch.device(device)))
    fake_imgs = celeb_generator(fixed_noise).detach().cpu()
    with torch.no_grad():
        grid = torchvision.utils.make_grid(fake_imgs, nrow=8, normalize=True)
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)  # channel dim should be last
        plt.matshow(grid_np)
        plt.show()

test_generator(64, celeb_generator, '/home/yashar/git/AD-with-GANs/data/DS5_celebA_bald/generator.pkl')
