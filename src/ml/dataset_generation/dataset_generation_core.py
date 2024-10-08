import sys

from src.ml.tools.utils import is_stylegan_dataset

sys.path.append('ml/models/StyleGAN')

import csv
import os
import shutil

import torch
import torch.optim as optim
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.ml.dataset_generation.ano_dataset import AnoDataset
from src.ml.latent_direction_explorer import LatentDirectionExplorer
from src.ml.latent_space_mapper import LatentSpaceMapper
from src.ml.models.base.beta_vae64 import BetaVAE64
from src.ml.models.base.discriminator import Discriminator
from src.ml.models.base.generator import Generator


def get_dataloader(dataset_folder, batch_size, transform, nrows=0, shuffle=True, unpolluted=False):
    ano_dataset = AnoDataset(
        root_dir=dataset_folder,
        transform=transform,
        nrows=nrows,
        unpolluted=unpolluted
    )

    return torch.utils.data.DataLoader(ano_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def generate_dataset(root_dir, temp_directory, dataset_name, generate_normals, generate_anomalies, ano_fraction):
    print('GENERATING BASE DATASET')
    dataset_root_folder = os.path.join(root_dir, dataset_name)
    dataset_folder = os.path.join(dataset_root_folder, 'dataset_raw')
    csv_path = os.path.join(dataset_folder, "ano_dataset.csv")

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


def equalize_image_sizes(final_image_size, root_dir, dataset_name, num_channels):
    dataset_folder = os.path.join(root_dir, dataset_name, 'dataset_raw')
    background_color = (0, 0, 0) if num_channels == 3 else 0
    for file in os.listdir(dataset_folder):
        if file.endswith(".png"):
            im = Image.open(os.path.join(dataset_folder, file))
            width, height = im.size
            if width == height:
                im = im.resize((final_image_size, final_image_size))
            elif width > height:
                result = Image.new(im.mode, (width, width), background_color)
                result.paste(im, (0, (width - height) // 2))
                im = result.resize((final_image_size, final_image_size))
            else:
                result = Image.new(im.mode, (height, height), background_color)
                result.paste(im, ((height - width) // 2, 0))
                im = result.resize((final_image_size, final_image_size))

            im.save(os.path.join(dataset_folder, file))


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
                       num_channels, batch_size,
                       device, learning_rate, transform=None, discriminator=None, generator=None, num_imgs=None,
                       save_checkpoint_every_n_epoch=10, initial_state_generator=None,
                       initial_state_discriminator=None, unpolluted=False):
    print('TRAINING GAN')
    dataset_folder = os.path.join(root_dir, dataset_name)
    dataset_raw_folder = os.path.join(dataset_folder, 'dataset_raw')
    checkpoint_folder = os.path.join(root_dir, '..', 'checkpoints', dataset_name)
    if os.path.exists(checkpoint_folder): shutil.rmtree(checkpoint_folder)

    if not generator:
        generator = Generator(z_dim=size_z,
                              num_feature_maps=num_feature_maps_g,
                              num_color_channels=num_channels).to(device)

    if initial_state_generator is not None and initial_state_discriminator is not None:
        generator.load_state_dict(torch.load(initial_state_generator, map_location=torch.device(device)))
        discriminator.load_state_dict(torch.load(initial_state_discriminator, map_location=torch.device(device)))
    else:
        generator.apply(weights_init)

    if not discriminator:
        discriminator = Discriminator(num_feature_maps=num_feature_maps_d,
                                      num_color_channels=num_channels).to(device)

    dataloader = get_dataloader(dataset_folder=dataset_raw_folder,
                                batch_size=batch_size,
                                transform=transform,
                                nrows=num_imgs,
                                unpolluted=unpolluted)
    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.
    adam_beta1 = 0.5

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(adam_beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(adam_beta1, 0.999))
    g_losses = []
    d_losses = []
    iterations = 0

    # scheduler_step_size = num_epochs // 5
    # scheduler_gamma = 0.5
    # scheduler_g = StepLR(optimizer_g, scheduler_step_size, gamma=scheduler_gamma)
    # scheduler_d = StepLR(optimizer_d, scheduler_step_size, gamma=scheduler_gamma)

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, (real_image, image_label, _) in enumerate(dataloader, 0):
            # Discriminator
            discriminator.zero_grad()

            bs = real_image.shape[0]
            real_image = real_image.to(device)
            label = torch.full((bs,), real_label, dtype=torch.float, device=device)
            output, _ = discriminator(real_image)
            output = output.view(-1)
            loss_d_real = criterion(output, label)
            loss_d_real.backward()
            d_x = output.mean().item()

            noise = torch.randn(bs, size_z, 1, 1, device=device)
            fake_images = generator(noise)
            label.fill_(fake_label)
            output, _ = discriminator(fake_images.detach())
            output = output.view(-1)
            loss_d_fake = criterion(output, label)

            loss_d_fake.backward()
            d_g_z1 = output.mean().item()
            loss_d = loss_d_real + loss_d_fake
            optimizer_d.step()
            # scheduler_d.step()

            # Generator
            generator.zero_grad()

            label.fill_(real_label)
            output, _ = discriminator(fake_images)
            output = output.view(-1)
            loss_g = criterion(output, label)
            loss_g.backward()
            d_g_z2 = output.mean().item()
            optimizer_g.step()
            # scheduler_g.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())
            iterations += 1

        # print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tLR: %.10f'
        #       % (epoch + 1, num_epochs, loss_d.item(), loss_g.item(), d_x, d_g_z1, d_g_z2, scheduler_g.get_last_lr()[0]))

        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch + 1, num_epochs, loss_d.item(), loss_g.item(), d_x, d_g_z1, d_g_z2))

        if epoch != 0 and epoch % save_checkpoint_every_n_epoch == 0:
            save_gan_checkpoint(checkpoint_folder, size_z, discriminator, epoch, iterations, generator, device)

    generator_filename = 'generator_unpolluted' if unpolluted else 'generator'
    discriminator_filename = 'discriminator_unpolluted' if unpolluted else 'discriminator'
    save_gan_models(dataset_folder,
                    discriminator,
                    generator,
                    generator_filename=generator_filename,
                    discriminator_filename=discriminator_filename)

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def save_gan_models(dataset_folder,
                    discriminator,
                    generator,
                    generator_filename='generator',
                    discriminator_filename='discriminator'):
    print("Saving Generator and Discriminator...")
    torch.save(generator.state_dict(), os.path.join(dataset_folder, f'{generator_filename}.pkl'))
    torch.save(generator, os.path.join(dataset_folder, f'{generator_filename}_model.pkl'))
    torch.save(discriminator.state_dict(), os.path.join(dataset_folder, f'{discriminator_filename}.pkl'))
    torch.save(discriminator, os.path.join(dataset_folder, f'{discriminator_filename}_model.pkl'))


def save_gan_checkpoint(checkpoint_folder, size_z, discriminator, epoch, iteration, generator, device):
    print("Saving GAN Checkpoints...")
    os.makedirs(checkpoint_folder, exist_ok=True)
    filename_state = f"epoch_{epoch}_iteration_{iteration}"
    generator_filename = f'{filename_state}_generator.pkl'
    generator_model_filename = f'{filename_state}_generator_model.pkl'
    discriminator_filename = f'{filename_state}_discriminator.pkl'
    discriminator_model_filename = f'{filename_state}_discriminator_model.pkl'
    torch.save(generator.state_dict(),
               os.path.join(checkpoint_folder, generator_filename))
    torch.save(generator, os.path.join(checkpoint_folder, generator_model_filename))
    torch.save(discriminator.state_dict(),
               os.path.join(checkpoint_folder, discriminator_filename))
    torch.save(discriminator, os.path.join(checkpoint_folder, discriminator_model_filename))

    # test_generator_and_show_plot(128, size_z, generator, os.path.join(checkpoint_folder, generator_filename), device)
    test_generator_and_save_plot(128, size_z, generator, os.path.join(checkpoint_folder, generator_filename), device,
                                 filename=os.path.join(checkpoint_folder, f'{filename_state}_generated_images.png'))


def train_direction_matrix(root_dir,
                           dataset_name,
                           direction_count,
                           steps,
                           device,
                           direction_batch_size=1,
                           z_dim=100,
                           num_channels=1,
                           use_bias=True,
                           generator=None,
                           reconstructor=None,
                           direction_train_shift_scale=6.0):
    print('TRAINING DIRECTION MATRIX')
    dataset_root_folder = os.path.join(root_dir, dataset_name)
    direction_matrices_folder = os.path.join(dataset_root_folder, 'direction_matrices')

    os.makedirs(dataset_root_folder, exist_ok=True)
    os.makedirs(direction_matrices_folder, exist_ok=True)

    trainer = LatentDirectionExplorer(z_dim=z_dim,
                                      directions_count=direction_count,
                                      bias=use_bias,
                                      device=device,
                                      direction_batch_size=direction_batch_size,
                                      num_channels=num_channels,
                                      saved_models_path=direction_matrices_folder,
                                      generator=generator,
                                      reconstructor=reconstructor,
                                      is_stylegan=is_stylegan_dataset(dataset_name),
                                      direction_train_shift_scale=direction_train_shift_scale)
    if not generator:
        trainer.load_generator(os.path.join(dataset_root_folder, 'generator.pkl'))

    b = 'bias' if use_bias else 'nobias'
    trainer.train_and_save(filename=f'direction_matrix_steps_{steps}_{b}_k_{direction_count}.pkl', num_steps=steps)
    shutil.rmtree(os.path.join(direction_matrices_folder, 'cp'))


def load_gans(root_dir, dataset_name, size_z, num_feature_maps_g, num_feature_maps_d, num_color_channels, device):
    generator_name = 'generator.pkl'
    generator_name_unpolluted = 'generator_unpolluted.pkl'
    discriminator_name = 'discriminator.pkl'
    discriminator_name_unpolluted = 'discriminator_unpolluted.pkl'

    generator = Generator(z_dim=size_z,
                          num_feature_maps=num_feature_maps_g,
                          num_color_channels=num_color_channels).to(device)
    discriminator = Discriminator(num_feature_maps=num_feature_maps_d,
                                  num_color_channels=num_color_channels).to(device)
    generator_unpolluted = Generator(z_dim=size_z,
                                     num_feature_maps=num_feature_maps_g,
                                     num_color_channels=num_color_channels).to(device)
    discriminator_unpolluted = Discriminator(num_feature_maps=num_feature_maps_d,
                                             num_color_channels=num_color_channels).to(device)

    generator.load_state_dict(
        torch.load(os.path.join(root_dir, dataset_name, generator_name), map_location=torch.device(device)))
    discriminator.load_state_dict(
        torch.load(os.path.join(root_dir, dataset_name, discriminator_name), map_location=torch.device(device)))

    generator_unpolluted.load_state_dict(
        torch.load(os.path.join(root_dir, dataset_name, generator_name_unpolluted),
                   map_location=torch.device(device)))
    discriminator_unpolluted.load_state_dict(
        torch.load(os.path.join(root_dir, dataset_name, discriminator_name_unpolluted),
                   map_location=torch.device(device)))

    return generator, discriminator, generator_unpolluted, discriminator_unpolluted


def load_gan_unpolluted(root_dir, dataset_name, size_z, num_feature_maps_g, num_feature_maps_d, num_color_channels,
                        device):
    generator = Generator(z_dim=size_z,
                          num_feature_maps=num_feature_maps_g,
                          num_color_channels=num_color_channels).to(device)
    discriminator = Discriminator(num_feature_maps=num_feature_maps_d,
                                  num_color_channels=num_color_channels).to(device)

    generator.load_state_dict(
        torch.load(os.path.join(root_dir, dataset_name, "generator_unpolluted.pkl"), map_location=torch.device(device)))
    discriminator.load_state_dict(
        torch.load(os.path.join(root_dir, dataset_name, "discriminator_unpolluted.pkl"),
                   map_location=torch.device(device)))

    return generator, discriminator


def train_beta_vae(device, root_dir, dataset_name, num_color_channels, num_epochs, batch_size):
    print('TRAINING BETA VARIATIONAL AUTOENCODER FOR VALIDATION')
    dataset_folder = os.path.join(root_dir, dataset_name)
    dataset_raw_folder = os.path.join(dataset_folder, 'dataset_raw')
    checkpoint_folder = os.path.join(root_dir, '..', 'checkpoints', dataset_name)
    if os.path.exists(checkpoint_folder): shutil.rmtree(checkpoint_folder)

    vae = BetaVAE64(device=device, num_color_channels=num_color_channels, num_epochs=num_epochs)

    transform = transforms.Compose([transforms.ToTensor()])
    dataloader = get_dataloader(dataset_folder=dataset_raw_folder,
                                batch_size=batch_size,
                                transform=transform)

    vae.train_model(dataloader)
    torch.save(vae, os.path.join(dataset_folder, 'vae_model.pkl'))


def create_latent_space_dataset(root_dir,
                                dataset_name,
                                size_z,
                                num_feature_maps_g,
                                num_feature_maps_d,
                                num_channels,
                                device,
                                n_latent_space_search_iterations=100000,
                                generator=None,
                                discriminator=None,
                                generator_unpolluted=None,
                                discriminator_unpolluted=None,
                                transform=None,
                                num_images=0,
                                start_with_image_number=0,
                                max_retries=3,
                                retry_threshold=0.06,
                                only_consider_anos=False,
                                retry_check_after_iter=5000,
                                learning_rate=0.001,
                                print_every_n_iters=5000,
                                draw_images=False,
                                use_discriminator_for_latent_space_mapping=True,
                                stylegan=False):
    print('MAPPING LATENT SPACE POINTS')
    dataset_folder = os.path.join(root_dir, dataset_name, 'dataset')
    dataset_raw_folder = os.path.join(root_dir, dataset_name, 'dataset_raw')
    if os.path.exists(dataset_folder) and start_with_image_number == 0:
        i = input("Dataset already exists. Do you want to overwrite it? Press y if yes")
        if i == 'y':
            shutil.rmtree(dataset_folder)
        else:
            print("Cancelling dataset generation")
            return

    os.makedirs(dataset_folder, exist_ok=True)
    csv_path = os.path.join(dataset_folder, "latent_space_mappings.csv")
    dataset = get_dataloader(dataset_folder=dataset_raw_folder,
                             batch_size=1,
                             transform=transform,
                             nrows=num_images,
                             shuffle=True)

    if not generator and not discriminator:
        generator, discriminator, generator_unpolluted, discriminator_unpolluted = load_gans(root_dir=root_dir,
                                                                                             dataset_name=dataset_name,
                                                                                             size_z=size_z,
                                                                                             num_feature_maps_g=num_feature_maps_g,
                                                                                             num_feature_maps_d=num_feature_maps_d,
                                                                                             num_color_channels=num_channels,
                                                                                             device=device)

    if not stylegan:
        generator_name = 'generator.pkl'
        generator_name_unpolluted = 'generator_unpolluted.pkl'
        discriminator_name = 'discriminator.pkl'
        discriminator_name_unpolluted = 'discriminator_unpolluted.pkl'
        generator.load_state_dict(
            torch.load(os.path.join(root_dir, dataset_name, generator_name), map_location=torch.device(device)))
        generator_unpolluted.load_state_dict(
            torch.load(os.path.join(root_dir, dataset_name, generator_name_unpolluted),
                       map_location=torch.device(device)))

        discriminator.load_state_dict(
            torch.load(os.path.join(root_dir, dataset_name, discriminator_name), map_location=torch.device(device)))
        discriminator_unpolluted.load_state_dict(
            torch.load(os.path.join(root_dir, dataset_name, discriminator_name_unpolluted),
                       map_location=torch.device(device)))

    generator.eval()
    discriminator.eval()

    os.makedirs(dataset_folder, exist_ok=True)
    lsm: LatentSpaceMapper = LatentSpaceMapper(generator=generator,
                                               discriminator=discriminator,
                                               device=device,
                                               stylegan=stylegan)
    lsm_unpolluted: LatentSpaceMapper = LatentSpaceMapper(generator=generator_unpolluted,
                                                          discriminator=discriminator_unpolluted,
                                                          device=device,
                                                          stylegan=stylegan)
    mapped_images = []
    cp_counter = 0
    counter = len(dataset) - start_with_image_number - 1

    i = start_with_image_number
    retry_counter = 0
    iterator = iter(dataset)
    if start_with_image_number == 0:
        add_line_to_csv(csv_path=csv_path, entries=["filename",
                                                    "label",
                                                    "original_filename",
                                                    "reconstruction_loss",
                                                    "reconstruction_loss_unpolluted"])
        data_point, data_label, orig_filename = next(iterator)

    for i in range(start_with_image_number):
        data_point, data_label, orig_filename = next(iterator)

    while counter > 0:
        if data_label.item() is False and only_consider_anos:
            counter -= 1
            data_point, data_label, orig_filename = next(iterator)
            continue

        print(f"{counter} images left")
        print(f"Label: {data_label.item()}")

        mapped_z, reconstruction_loss, retry = lsm.map_image_to_point_in_latent_space(image=data_point,
                                                                                      n_iterations=n_latent_space_search_iterations,
                                                                                      retry_check_after_iter=retry_check_after_iter,
                                                                                      learning_rate=learning_rate,
                                                                                      print_every_n_iters=print_every_n_iters,
                                                                                      retry_threshold=retry_threshold,
                                                                                      use_discriminator_for_latent_space_mapping=use_discriminator_for_latent_space_mapping,
                                                                                      stylegan=stylegan)

        mapped_z_unpolluted, reconstruction_loss_unpolluted, _ = lsm_unpolluted.map_image_to_point_in_latent_space(
            image=data_point,
            n_iterations=n_latent_space_search_iterations,
            retry_check_after_iter=retry_check_after_iter,
            learning_rate=learning_rate,
            print_every_n_iters=100000,
            retry_threshold=100000,
            use_discriminator_for_latent_space_mapping=use_discriminator_for_latent_space_mapping,
            stylegan=stylegan)

        if retry:
            if retry_counter == max_retries:
                retry_counter = 0
                i += 1
                counter -= 1
                print("Retry Limit reached. Moving on to next sample")
                print("Could not map this image")
                plot_image(data_point[0])
                data_point, data_label, orig_filename = next(iterator)
                print('-----------------------')
                continue
            else:
                retry_counter += 1
                print(
                    f"Could not find optimal region within the defined iteration count. Retry ({retry_counter}) with another random z...")
                continue

        retry_counter = 0

        if draw_images:
            mapped_img = generator(mapped_z)[0]
            mapped_img_unpolluted = generator_unpolluted(mapped_z_unpolluted)[0]
            plot_image(data_point[0])
            plot_image(mapped_img)
            plot_image(mapped_img_unpolluted)

        mapped_images.append(mapped_z)
        add_line_to_csv(csv_path=csv_path,
                        entries=[
                            f'mapped_z_{counter}.pt',
                            data_label.item(),
                            orig_filename[0],
                            reconstruction_loss,
                            reconstruction_loss_unpolluted
                        ])
        torch.save(mapped_z, os.path.join(dataset_folder, f'mapped_z_{counter}.pt'))
        cp_counter += 1

        i += 1
        counter -= 1
        data_point, data_label, orig_filename = next(iterator)

    print('All images in dataset were mapped')


def test_generator(num, size_z, g, g_path, device):
    fixed_noise = torch.randn(num, size_z, 1, 1, device=device)
    g.load_state_dict(torch.load(g_path, map_location=torch.device(device)))
    fake_imgs = g(fixed_noise).detach().cpu()
    with torch.no_grad():
        grid = torchvision.utils.make_grid(fake_imgs, nrow=16, normalize=True)
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)  # channel dim should be last
        plt.matshow(grid_np)
        plt.axis("off")
        return plt


def test_generator_and_show_plot(num, size_z, g, g_path, device):
    test_generator(num, size_z, g, g_path, device)
    plt.show(block=False)
    plt.close()


def test_generator_and_save_plot(num, size_z, g, g_path, device, filename):
    test_generator(num, size_z, g, g_path, device)
    plt.savefig(filename)
    plt.close()


def plot_image(img):
    img = (img * 0.5) + 0.5
    img = img.cpu().detach().numpy().transpose(1, 2, 0)
    imshow(img)
    plt.show(block=False)
