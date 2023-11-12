from abc import ABC, abstractmethod

import torch
import torchvision.transforms as transforms

from src.ml.datasets.generate_dataset import create_latent_space_dataset, train_direction_matrix, generate_dataset, \
    train_and_save_gan, equalize_image_sizes
from src.ml.models.base.discriminator_master import DiscriminatorMaster
from src.ml.models.base.generator_master import GeneratorMaster
from src.ml.models.base.reconstructor import Reconstructor


class AbstractDatasetGenerator(ABC):
    def __init__(self,
                 dataset_name,
                 root_dir='../data',
                 temp_directory='../data_temp',
                 num_color_channels=3,
                 batch_size=512,
                 image_size=64,
                 size_z=100,
                 learning_rate=0.001,
                 gan_num_epochs=2500,
                 num_feature_maps_g=64,
                 num_feature_maps_d=64,
                 num_imgs=0,
                 save_checkpoint_every_n_epoch=100,
                 directions_count=30,
                 direction_train_steps=2500,
                 use_bias=True,
                 n_iterations=5000,
                 max_retries=5,
                 retry_threshold=0.1,
                 only_consider_anos=True,
                 retry_check_after_iter=2500,
                 start_learning_rate=0.0001,
                 print_every_n_iters=2500,
                 draw_images=True):
        self.root_dir = root_dir
        self.temp_directory = temp_directory
        self.dataset_name = dataset_name
        self.num_color_channels = num_color_channels
        self.batch_size = batch_size
        self.image_size = image_size
        self.size_z = size_z
        self.learning_rate = learning_rate
        self.gan_num_epochs = gan_num_epochs
        self.num_feature_maps_g = num_feature_maps_g
        self.num_feature_maps_d = num_feature_maps_d
        self.num_imgs = num_imgs
        self.save_checkpoint_every_n_epoch = save_checkpoint_every_n_epoch
        self.directions_count = directions_count
        self.direction_train_steps = direction_train_steps
        self.use_bias = use_bias
        self.n_iterations = n_iterations
        self.max_retries = max_retries
        self.retry_threshold = retry_threshold
        self.only_consider_anos = only_consider_anos
        self.retry_check_after_iter = retry_check_after_iter
        self.start_learning_rate = start_learning_rate
        self.print_every_n_iters = print_every_n_iters
        self.draw_images = draw_images

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ) if self.num_color_channels == 3 else transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )

        self.generator = GeneratorMaster(size_z=self.size_z, num_feature_maps=self.num_feature_maps_g,
                                         num_color_channels=self.num_color_channels,
                                         dropout_rate=0.1).to(self.device)

        self.discriminator = DiscriminatorMaster(num_feature_maps=self.num_feature_maps_d,
                                                 num_color_channels=self.num_color_channels,
                                                 dropout_rate=0.1).to(self.device)

        self.reconstructor = Reconstructor(directions_count=self.directions_count,
                                           width=2).to(self.device)

    @abstractmethod
    def generate_normals(self, dataset_folder, csv_path, temp_directory):
        pass

    @abstractmethod
    def generate_anomalies(self, dataset_folder, csv_path, temp_directory, ano_fraction):
        pass

    def run_generate_dataset(self):
        generate_dataset(root_dir=self.root_dir,
                         temp_directory=self.temp_directory,
                         dataset_name=self.dataset_name,
                         generate_normals=self.generate_normals,
                         generate_anomalies=self.generate_anomalies,
                         ano_fraction=0.1)

    def run_equalize_image_sizes(self):
        equalize_image_sizes(final_image_size=self.image_size,
                             root_dir=self.root_dir,
                             dataset_name=self.dataset_name)

    def run_train_and_save_gan(self):
        train_and_save_gan(root_dir=self.root_dir,
                           dataset_name=self.dataset_name,
                           size_z=self.size_z,
                           num_epochs=self.gan_num_epochs,
                           num_feature_maps_g=self.num_feature_maps_g,
                           num_feature_maps_d=self.num_feature_maps_d,
                           num_color_channels=self.num_color_channels,
                           batch_size=self.batch_size,
                           device=self.device,
                           learning_rate=self.learning_rate,
                           generator=self.generator,
                           discriminator=self.discriminator,
                           transform=self.transform,
                           num_imgs=self.num_imgs,
                           save_checkpoint_every_n_epoch=self.save_checkpoint_every_n_epoch)

    def run_train_direction_matrix(self):
        train_direction_matrix(root_dir=self.root_dir,
                               dataset_name=self.dataset_name,
                               direction_count=self.directions_count,
                               steps=self.direction_train_steps,
                               device=self.device,
                               use_bias=self.use_bias,
                               generator=self.generator,
                               reconstructor=self.reconstructor)

    def run_create_latent_space_dataset(self):
        create_latent_space_dataset(root_dir=self.root_dir,
                                    transform=self.transform,
                                    dataset_name=self.dataset_name,
                                    size_z=self.size_z,
                                    num_feature_maps_g=self.num_feature_maps_g,
                                    num_feature_maps_d=self.num_feature_maps_d,
                                    num_color_channels=self.num_color_channels,
                                    device=self.device,
                                    n_iterations=self.n_iterations,
                                    generator=self.generator,
                                    discriminator=self.discriminator,
                                    max_retries=self.max_retries,
                                    retry_threshold=self.retry_threshold,
                                    only_consider_anos=self.only_consider_anos,
                                    retry_check_after_iter=self.retry_check_after_iter,
                                    learning_rate=self.start_learning_rate,
                                    print_every_n_iters=self.print_every_n_iters,
                                    draw_images=self.draw_images)

    def run(self):
        self.run_generate_dataset()
        self.run_equalize_image_sizes()
        self.run_train_and_save_gan()
        self.run_train_direction_matrix()
        self.run_create_latent_space_dataset()
