import math
import os
import shutil
import csv
import time

import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils

from src.ml.latent_direction_explorer import LatentDirectionExplorer
from src.ml.latent_space_mapper import LatentSpaceMapper
from src.ml.models.discriminator import Discriminator
from src.ml.models.generator import Generator
from src.ml.datasets.ano_mnist import AnoMNIST


def get_dataloader(dataset_folder, batch_size, transform=None, nrows=None):
    if not transform:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5,), std=(.5,))
        ])

    ano_mnist_dataset = AnoMNIST(
        root_dir=dataset_folder,
        transform=transform,
        nrows=nrows
    )

    return torch.utils.data.DataLoader(ano_mnist_dataset, batch_size=batch_size, shuffle=True)


def generate_dataset(root_dir, temp_directory, dataset_name, generate_normals, generate_anomalies, ano_fraction):
    print('GENERATING BASE DATASET')
    dataset_root_folder = os.path.join(root_dir, dataset_name)
    dataset_folder = os.path.join(dataset_root_folder, 'dataset_raw')
    csv_path = os.path.join(dataset_folder, "anomnist_dataset.csv")

    if os.path.exists(dataset_root_folder):
        i = input("Dataset already exists. Do you want to overwrite it? Press y if yes")
        if i == 'y':
            shutil.rmtree(dataset_root_folder)
        else:
            print("Cancelling dataset generation")
            return

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(dataset_root_folder, exist_ok=True)
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(temp_directory, exist_ok=True)

    add_line_to_csv(csv_path, ['filename', 'anomaly'])

    generate_normals(dataset_folder, csv_path, temp_directory)
    generate_anomalies(dataset_folder, csv_path, temp_directory, ano_fraction)


def add_line_to_csv(csv_path, entries: list[str]):
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(entries)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_and_save_gan(root_dir, dataset_name, size_z, num_epochs, num_feature_maps_g, num_feature_maps_d,
                       num_color_channels, batch_size,
                       device, learning_rate, transform=None, discriminator=None, generator=None, num_imgs=None):
    print('TRAINING GAN')
    dataset_folder = os.path.join(root_dir, dataset_name)
    dataset_raw_folder = os.path.join(dataset_folder, 'dataset_raw')
    checkpoint_folder = os.path.join(root_dir, '..', 'checkpoints', dataset_name)

    if not generator:
        generator = Generator(size_z=size_z,
                              num_feature_maps=num_feature_maps_g,
                              num_color_channels=num_color_channels).to(device)
    generator.apply(weights_init)

    if not discriminator:
        discriminator = Discriminator(num_feature_maps=num_feature_maps_d,
                                      num_color_channels=num_color_channels).to(device)

    dataloader = get_dataloader(dataset_folder=dataset_raw_folder, batch_size=batch_size, transform=transform, nrows=num_imgs)
    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.
    adam_beta1 = 0.5

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(adam_beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(adam_beta1, 0.999))
    g_losses = []
    d_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader, 0):
            # Discriminator
            discriminator.zero_grad()

            bs = real_images.shape[0]
            real_images = real_images.to(device)
            label = torch.full((bs,), real_label, dtype=torch.float, device=device)
            # output, _ = discriminator(real_images)
            output, _ = discriminator(real_images)
            output = output.view(-1)
            loss_d_real = criterion(output, label)
            loss_d_real.backward()
            d_x = output.mean().item()

            noise = torch.randn(bs, size_z, 1, 1, device=device)
            fake_images = generator(noise)
            label.fill_(fake_label)
            # output, _ = discriminator(fake_images.detach())
            output, _ = discriminator(fake_images.detach())
            output = output.view(-1)
            loss_d_fake = criterion(output, label)

            loss_d_fake.backward()
            d_g_z1 = output.mean().item()
            loss_d = loss_d_real + loss_d_fake
            optimizer_d.step()

            # Generator
            generator.zero_grad()

            label.fill_(real_label)
            output, _ = discriminator(fake_images)
            output = output.view(-1)
            loss_g = criterion(output, label)
            loss_g.backward()
            d_g_z2 = output.mean().item()
            optimizer_g.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())

            # if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            #     with torch.no_grad():
            #         fake = generator(fixed_noise).detach().cpu()
            #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1

        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch + 1, num_epochs, loss_d.item(), loss_g.item(), d_x, d_g_z1, d_g_z2))

        if epoch == 1 or epoch % 5 == 0:
            save_gan_checkpoint(checkpoint_folder, discriminator, epoch, generator)

    save_gan_models(dataset_folder, discriminator, generator)
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def save_gan_models(dataset_folder, discriminator, generator):
    print("Saving Generator and Discriminator...")
    torch.save(generator.state_dict(), os.path.join(dataset_folder, f'generator.pkl'))
    torch.save(discriminator.state_dict(), os.path.join(dataset_folder, f'discriminator.pkl'))


def save_gan_checkpoint(checkpoint_folder, discriminator, epoch, generator):
    print("Saving GAN Checkpoints...")
    os.makedirs(checkpoint_folder, exist_ok=True)
    timestamp = time.time()
    torch.save(generator.state_dict(),
               os.path.join(checkpoint_folder, f'generator_epoch_{epoch}_{timestamp}.pkl'))
    torch.save(discriminator.state_dict(),
               os.path.join(checkpoint_folder, f'discriminator_epoch_{epoch}_{timestamp}.pkl'))


def train_direction_matrix(root_dir, dataset_name, direction_count, steps, device, use_bias=True, generator=None,
                           reconstructor=None):
    print('TRAINING DIRECTION MATRIX')
    dataset_root_folder = os.path.join(root_dir, dataset_name)
    direction_matrices_folder = os.path.join(dataset_root_folder, 'direction_matrices')

    os.makedirs(dataset_root_folder, exist_ok=True)
    os.makedirs(direction_matrices_folder, exist_ok=True)

    trainer = LatentDirectionExplorer(z_dim=100,
                                      directions_count=direction_count,
                                      bias=use_bias,
                                      device=device,
                                      saved_models_path=direction_matrices_folder,
                                      generator=generator,
                                      reconstructor=reconstructor)
    trainer.load_generator(os.path.join(dataset_root_folder, 'generator.pkl'))
    b = 'bias' if use_bias else 'nobias'
    trainer.train_and_save(filename=f'direction_matrix_steps_{steps}_{b}_k_{direction_count}.pkl', num_steps=steps)
    shutil.rmtree(os.path.join(direction_matrices_folder, 'cp'))


def load_gan(root_dir, dataset_name, size_z, num_feature_maps_g, num_feature_maps_d, num_color_channels, device):
    generator = Generator(size_z=size_z,
                          num_feature_maps=num_feature_maps_g,
                          num_color_channels=num_color_channels).to(device)
    discriminator = Discriminator(num_feature_maps=num_feature_maps_d,
                                  num_color_channels=num_color_channels).to(device)

    generator.load_state_dict(
        torch.load(os.path.join(root_dir, dataset_name, "generator.pkl"), map_location=torch.device(device)))
    discriminator.load_state_dict(
        torch.load(os.path.join(root_dir, dataset_name, "discriminator.pkl"), map_location=torch.device(device)))

    return generator, discriminator


def create_latent_space_dataset(root_dir, dataset_name, size_z, num_feature_maps_g, num_feature_maps_d,
                                num_color_channels, device,  max_opt_iterations=100000, generator=None, discriminator=None, transform=None):
    print('MAPPING LATENT SPACE POINTS')
    dataset_folder = os.path.join(root_dir, dataset_name, 'dataset')
    dataset_raw_folder = os.path.join(root_dir, dataset_name, 'dataset_raw')
    if os.path.exists(dataset_folder):
        i = input("Dataset already exists. Do you want to overwrite it? Press y if yes")
        if i == 'y':
            shutil.rmtree(dataset_folder)
        else:
            print("Cancelling dataset generation")
            return

    os.makedirs(dataset_folder, exist_ok=True)
    csv_path = os.path.join(dataset_folder, "latent_space_mappings.csv")
    dataset = get_dataloader(dataset_folder=dataset_raw_folder, batch_size=1, transform=transform, nrows=1)

    if not generator and not discriminator:
        generator, discriminator = load_gan(root_dir=root_dir,
                                            dataset_name=dataset_name,
                                            size_z=size_z,
                                            num_feature_maps_g=num_feature_maps_g,
                                            num_feature_maps_d=num_feature_maps_d,
                                            num_color_channels=num_color_channels,
                                            device=device)

    generator.load_state_dict(
        torch.load(os.path.join(root_dir, dataset_name, "generator.pkl"), map_location=torch.device(device)))
    discriminator.load_state_dict(
        torch.load(os.path.join(root_dir, dataset_name, "discriminator.pkl"), map_location=torch.device(device)))


    os.makedirs(dataset_folder, exist_ok=True)
    add_line_to_csv(csv_path=csv_path, entries=["filename", "label", "reconstruction_loss"])

    lsm: LatentSpaceMapper = LatentSpaceMapper(generator=generator, discriminator=discriminator, device=device)
    mapped_images = []
    cp_counter = 0
    counter = len(dataset)

    i = 0
    retry_counter = 0
    iterator = iter(dataset)
    data_point, data_label = next(iterator)

    while counter > 0:
        print(f"{counter} images left")
        print(f"Label: {data_label.item()}")

        max_retries = 0
        opt_threshold = 0.001
        ignore_rules_below_threshold = 10000
        immediate_retry_threshold = 1000000

        plot_image(data_point[0])
        mapped_z, reconstruction_loss, retry = lsm.map_image_to_point_in_latent_space(image=data_point,
                                                                                      max_opt_iterations=max_opt_iterations,
                                                                                      plateu_threshold=0,
                                                                                      check_every_n_iter=5000,
                                                                                      learning_rate=0.0001,
                                                                                      print_every_n_iters=1000,
                                                                                      retry_after_n_iters=3000000,
                                                                                      ignore_rules_below_threshold=ignore_rules_below_threshold,
                                                                                      opt_threshold=opt_threshold,
                                                                                      immediate_retry_threshold=immediate_retry_threshold)
        if retry:
            if retry_counter == max_retries:
                retry_counter = 0
                i += 1
                counter -= 1
                print("Retry Limit reached. Moving on to next sample")
                data_point, data_label = next(iterator)
                print('-----------------------')
                continue
            else:
                retry_counter += 1
                print(
                    f"Could not find optimal region within the defined iteration count. Retry ({retry_counter}) with another random z...")
                continue

        retry_counter = 0

        mapped_img = generator(mapped_z)[0]
        plot_image(mapped_img)
        mapped_images.append(mapped_z)
        add_line_to_csv(csv_path=csv_path,
                        entries=[f'mapped_z_{counter}.pt', data_label.item(), math.floor(reconstruction_loss)])
        torch.save(mapped_z, os.path.join(dataset_folder, f'mapped_z_{counter}.pt'))
        cp_counter += 1

        i += 1
        counter -= 1
        data_point, data_label = next(iterator)

    print('All images in dataset were mapped')


def plot_image(img):
    img = (img * 0.5) + 0.5
    img = img.permute(1, 2, 0)
    imshow(img.detach().cpu().numpy())
    plt.show()
