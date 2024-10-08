import os
import shutil
import time

import torch
import torch.nn as nn

from src.ml.models.base.generator import Generator
from src.ml.models.base.matrix_a_linear import MatrixALinear
from src.ml.models.base.reconstructor import Reconstructor
from src.ml.tools.utils import generate_noise


class LatentDirectionExplorer:
    def __init__(self, z_dim,
                 directions_count,
                 device,
                 saved_models_path,
                 num_channels=1,
                 direction_batch_size=1,
                 direction_train_shift_scale=6.0,
                 bias=True,
                 generator=None,
                 reconstructor=None,
                 is_stylegan=False, ):
        super(LatentDirectionExplorer, self).__init__()
        self.min_shift = 0.5
        self.direction_train_shift_scale = direction_train_shift_scale
        self.matrix_a_lr = 0.002
        self.reconstructor_lr = 0.002
        self.label_weight = 1.0
        self.shift_weight = 0.25
        self.cross_entropy = nn.CrossEntropyLoss().to(device)
        self.saved_models_path = saved_models_path

        self.direction_batch_size = direction_batch_size
        self.z_dim = z_dim
        self.directions_count = directions_count
        self.device = device
        self.is_stylegan = is_stylegan

        # init Generator
        if not generator:
            self.g: Generator = Generator(z_dim=self.z_dim, num_feature_maps=64, num_color_channels=1).to(device)
        else:
            self.g = generator

        # init MatrixA
        self.matrix_a = MatrixALinear(input_dim=self.directions_count, bias=bias, output_dim=z_dim).to(device)

        # init Reconstructor
        if not reconstructor:
            self.reconstructor = (
                Reconstructor(directions_count=self.matrix_a.input_dim, num_channels=num_channels).to(device)
            )
        else:
            self.reconstructor = reconstructor

    def train_and_save(self, filename, num_steps=1000):
        # init optimizers for MatrixA, Reconstructor
        matrix_a_opt = torch.optim.Adam(self.matrix_a.parameters(), lr=self.matrix_a_lr)
        reconstructor_opt = torch.optim.Adam(self.reconstructor.parameters(), lr=self.reconstructor_lr)

        cp_folder = f"{self.saved_models_path}/cp"
        if os.path.exists(cp_folder):
            shutil.rmtree(cp_folder)
            os.makedirs(f"{cp_folder}")

        if not os.path.exists(cp_folder):
            os.makedirs(f"{cp_folder}")

        # start training loop
        for step in range(num_steps):
            if step % 100 == 0:
                print(f'Step {step} of {num_steps}')

            self.g.zero_grad()
            self.matrix_a.zero_grad()
            self.reconstructor.zero_grad()

            # cast random noise z
            if self.is_stylegan:
                z = torch.randn(self.direction_batch_size, self.z_dim, device=self.device)
            else:
                z = generate_noise(batch_size=self.direction_batch_size, z_dim=self.z_dim, device=self.device)

            # generate shifts
            # cast random integer that represents the k^th column  --> e_k
            target_indices, shifts, basis_shift = self.__make_shifts(self.matrix_a.input_dim, self.direction_batch_size)
            shift = self.matrix_a(basis_shift)

            # generate images --> from z and from z + A(epsilon * e_k)
            images = self.g(z)
            images_shifted = self.g.gen_shifted(z, shift)

            logits, shift_prediction = self.reconstructor(images, images_shifted)
            logit_loss = self.label_weight * self.cross_entropy(logits.to(self.device), target_indices.to(self.device))
            shift_loss = self.shift_weight * torch.mean(torch.abs(shift_prediction - shifts))

            # total loss
            loss = logit_loss + shift_loss
            loss.backward()

            matrix_a_opt.step()
            reconstructor_opt.step()

        # save a and R
        self.__save_models(filename)

        # display image from A(x) with shift epsilon
        print(self.matrix_a)

    def load_generator(self, path):
        self.g.load_state_dict(torch.load(path, map_location=torch.device(self.device)))

    def set_generator(self, g):
        self.g = g

    def __make_shifts(self, latent_dim, batch_size):
        target_indices = torch.randint(0, self.directions_count, [batch_size])

        # Casting from uniform distribution
        # See https://github.com/anvoynov/GANLatentDiscovery/blob/5ca8d67bce8dcb9a51de07c98e2d3a0d6ab69fe3/trainer.py#L75
        shifts = 2.0 * torch.rand(target_indices.shape, device=self.device) - 1.0

        shifts = self.direction_train_shift_scale * shifts
        shifts[(shifts < self.min_shift) & (shifts > 0)] = self.min_shift
        shifts[(shifts > -self.min_shift) & (shifts < 0)] = -self.min_shift

        z_shift = torch.zeros([self.direction_batch_size] + [latent_dim], device=self.device)
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift

    def __save_models(self, filename):
        print("Saving models...")
        if self.is_stylegan:
            torch.save(self.matrix_a, f'{self.saved_models_path}/{filename}')
        else:
            torch.save(self.matrix_a.state_dict(), f'{self.saved_models_path}/{filename}')

    def __save_checkpoint(self, iteration):
        torch.save(self.matrix_a.state_dict(), f'{self.saved_models_path}/cp/matrix_a_{time.time()}_{iteration}.pkl')
        torch.save(self.reconstructor.state_dict(),
                   f'{self.saved_models_path}/cp/reconstructor_{time.time()}_{iteration}.pkl')
