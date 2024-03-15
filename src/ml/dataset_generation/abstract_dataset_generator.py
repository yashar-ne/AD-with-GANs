import os
from abc import ABC, abstractmethod

import torch
import torchvision.transforms as transforms

from src.ml.dataset_generation.dataset_generation_core import create_latent_space_dataset, train_direction_matrix, \
    generate_dataset, \
    train_and_save_gan, equalize_image_sizes, test_generator_and_show_plot, train_beta_vae
from src.ml.models.StyleGAN import dnnlib, legacy
from src.ml.models.StyleGAN.ano_detection.stylegan_discriminator_wrapper import StyleGANDiscriminatorWrapper
from src.ml.models.StyleGAN.ano_detection.stylegan_generator_wrapper import StyleGANGeneratorWrapper
from src.ml.models.StyleGAN.ano_detection.stylegan_reconstructor import StyleGANReconstructor
from src.ml.models.base.discriminator_master import DiscriminatorMaster
from src.ml.models.base.generator_master import GeneratorMaster
from src.ml.models.base.reconstructor import Reconstructor
from src.ml.tools.utils import is_stylegan_dataset


class AbstractDatasetGenerator(ABC):
    def __init__(self,
                 dataset_name,
                 num_channels,
                 num_epochs,
                 num_imgs=0,
                 root_dir='../data',
                 temp_directory='../data_temp',
                 batch_size=512,
                 image_size=64,
                 size_z=100,
                 learning_rate=0.001,
                 num_feature_maps_g=64,
                 num_feature_maps_d=64,
                 save_checkpoint_every_n_epoch=100,
                 directions_count=30,
                 direction_batch_size=1,
                 direction_train_steps=2500,
                 direction_train_shift_scale=6.0,
                 use_bias=True,
                 n_latent_space_search_iterations=5000,
                 max_retries=5,
                 retry_threshold=0.1,
                 only_consider_anos=False,
                 retry_check_after_iter=2500,
                 start_learning_rate=0.0001,
                 print_every_n_iters=2500,
                 draw_images=False,
                 use_discriminator_for_latent_space_mapping=True,
                 stylegan=False):
        self.root_dir = root_dir
        self.temp_directory = temp_directory
        self.dataset_name = dataset_name
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.image_size = image_size
        self.size_z = size_z
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_feature_maps_g = num_feature_maps_g
        self.num_feature_maps_d = num_feature_maps_d
        self.num_imgs = num_imgs
        self.save_checkpoint_every_n_epoch = save_checkpoint_every_n_epoch
        self.directions_count = directions_count
        self.direction_batch_size = direction_batch_size
        self.direction_train_steps = direction_train_steps
        self.direction_train_shift_scale = direction_train_shift_scale
        self.use_bias = use_bias
        self.n_latent_space_search_iterations = n_latent_space_search_iterations
        self.max_retries = max_retries
        self.retry_threshold = retry_threshold
        self.only_consider_anos = only_consider_anos
        self.retry_check_after_iter = retry_check_after_iter
        self.start_learning_rate = start_learning_rate
        self.print_every_n_iters = print_every_n_iters
        self.draw_images = draw_images
        self.use_discriminator_for_latent_space_mapping = use_discriminator_for_latent_space_mapping
        self.stylegan = stylegan

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ) if self.num_channels == 3 else transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )

        if not is_stylegan_dataset(self.dataset_name):
            self.generator = GeneratorMaster(z_dim=self.size_z, num_feature_maps=self.num_feature_maps_g,
                                             num_channels=self.num_channels,
                                             dropout_rate=0.1).to(self.device)

            self.discriminator = DiscriminatorMaster(num_feature_maps=self.num_feature_maps_d,
                                                     num_channels=self.num_channels,
                                                     dropout_rate=0.1).to(self.device)

            self.generator_unpolluted = GeneratorMaster(z_dim=self.size_z, num_feature_maps=self.num_feature_maps_g,
                                                        num_channels=self.num_channels,
                                                        dropout_rate=0.1).to(self.device)

            self.discriminator_unpolluted = DiscriminatorMaster(num_feature_maps=self.num_feature_maps_d,
                                                                num_channels=self.num_channels,
                                                                dropout_rate=0.1).to(self.device)

            self.reconstructor = Reconstructor(directions_count=self.directions_count,
                                               num_channels=self.num_channels,
                                               width=2).to(self.device)

        else:
            models_path = os.path.join(self.root_dir, self.dataset_name, "models.pkl")
            with dnnlib.util.open_url(str(models_path)) as f:
                networks = legacy.load_network_pkl(f)
                self.generator = StyleGANGeneratorWrapper(networks['G_ema']).to(self.device)
                self.discriminator = StyleGANDiscriminatorWrapper(networks['D']).to(self.device)

            self.reconstructor = StyleGANReconstructor(directions_count=self.directions_count,
                                                       num_channels=self.num_channels,
                                                       width=2).to(self.device)

    @abstractmethod
    def generate_normals(self, dataset_folder, csv_path, temp_directory):
        pass

    @abstractmethod
    def generate_anomalies(self, dataset_folder, csv_path, temp_directory, ano_fraction):
        pass

    def run_generate_dataset(self, ano_fraction=0.1):
        generate_dataset(root_dir=self.root_dir,
                         temp_directory=self.temp_directory,
                         dataset_name=self.dataset_name,
                         generate_normals=self.generate_normals,
                         generate_anomalies=self.generate_anomalies,
                         ano_fraction=ano_fraction)

    def run_equalize_image_sizes(self):
        equalize_image_sizes(final_image_size=self.image_size,
                             root_dir=self.root_dir,
                             dataset_name=self.dataset_name,
                             num_channels=self.num_channels)

    def run_train_and_save_gan(self, display_generator_test=True, unpolluted=False):
        train_and_save_gan(root_dir=self.root_dir,
                           dataset_name=self.dataset_name,
                           size_z=self.size_z,
                           num_epochs=self.num_epochs,
                           num_feature_maps_g=self.num_feature_maps_g,
                           num_feature_maps_d=self.num_feature_maps_d,
                           num_channels=self.num_channels,
                           batch_size=self.batch_size,
                           device=self.device,
                           learning_rate=self.learning_rate,
                           generator=self.generator,
                           discriminator=self.discriminator,
                           transform=self.transform,
                           num_imgs=self.num_imgs,
                           save_checkpoint_every_n_epoch=self.save_checkpoint_every_n_epoch,
                           unpolluted=unpolluted)

        if display_generator_test:
            generator_name = 'generator' if not unpolluted else 'generator_unpolluted'
            test_generator_and_show_plot(128,
                                         self.size_z,
                                         self.generator,
                                         os.path.join(self.root_dir, self.dataset_name, f'{generator_name}.pkl'),
                                         self.device)

    def run_train_direction_matrix(self):
        train_direction_matrix(z_dim=self.size_z,
                               root_dir=self.root_dir,
                               dataset_name=self.dataset_name,
                               direction_count=self.directions_count,
                               direction_batch_size=self.direction_batch_size,
                               steps=self.direction_train_steps,
                               device=self.device,
                               use_bias=self.use_bias,
                               generator=self.generator,
                               reconstructor=self.reconstructor,
                               direction_train_shift_scale=self.direction_train_shift_scale)

    def run_train_beta_vae(self):
        train_beta_vae(device=self.device,
                       root_dir=self.root_dir,
                       dataset_name=self.dataset_name,
                       num_color_channels=self.num_channels,
                       num_epochs=self.num_epochs,
                       batch_size=self.batch_size)

    def run_create_latent_space_dataset(self):
        create_latent_space_dataset(root_dir=self.root_dir,
                                    transform=self.transform,
                                    dataset_name=self.dataset_name,
                                    size_z=self.size_z,
                                    num_feature_maps_g=self.num_feature_maps_g,
                                    num_feature_maps_d=self.num_feature_maps_d,
                                    num_channels=self.num_channels,
                                    device=self.device,
                                    n_latent_space_search_iterations=self.n_latent_space_search_iterations,
                                    generator=self.generator,
                                    generator_unpolluted=self.generator_unpolluted,
                                    discriminator=self.discriminator,
                                    discriminator_unpolluted=self.discriminator_unpolluted,
                                    max_retries=self.max_retries,
                                    retry_threshold=self.retry_threshold,
                                    only_consider_anos=self.only_consider_anos,
                                    retry_check_after_iter=self.retry_check_after_iter,
                                    learning_rate=self.start_learning_rate,
                                    print_every_n_iters=self.print_every_n_iters,
                                    draw_images=self.draw_images,
                                    use_discriminator_for_latent_space_mapping=self.use_discriminator_for_latent_space_mapping,
                                    stylegan=self.stylegan)

    def run(self, ano_fraction):
        self.run_generate_dataset(ano_fraction=ano_fraction)
        self.run_equalize_image_sizes()
        self.run_train_and_save_gan()
        self.run_train_direction_matrix()
        self.run_train_beta_vae()
        self.run_create_latent_space_dataset()
