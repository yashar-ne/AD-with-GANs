import torch
import torch.nn as nn

from src.models.generator import Generator
from src.models.matrix_a_linear import MatrixALinear
from src.models.reconstructor import Reconstructor
from src.tools.utils import make_noise


class Trainer:
    def __init__(self, z_dim, directions_count, latent_dim, min_shift, shift_scale, epsilon=6):
        super(Trainer, self).__init__()
        self.min_shift = min_shift
        self.shift_scale = shift_scale

        self.cross_entropy = nn.CrossEntropyLoss()
        self.label_weight = 1.0
        self.shift_weight = 0.25

        # init Generator
        self.g = Generator(size_z=z_dim, num_feature_maps=64, num_color_channels=1)

        # init MatrixA
        self.matrix_a = MatrixALinear(input_dim=directions_count, inner_dim=latent_dim, output_dim=latent_dim)

        # init Reconstructor
        self.reconstructor = Reconstructor(dim=self.matrix_a.input_dim)

    def load_models_from_checkpoints(self):
        print("loading models...")

    def make_shifts(self, latent_dim, directions_count, batch_size):
        target_indices = torch.randint(0, directions_count, [batch_size])
        shifts = 2.0 * torch.rand(target_indices.shape) - 1.0

        shifts = self.shift_scale * shifts
        shifts[(shifts < self.min_shift) & (shifts > 0)] = self.min_shift
        shifts[(shifts > -self.min_shift) & (shifts < 0)] = -self.min_shift

        z_shift = torch.zeros([self.p.batch_size] + latent_dim)
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift

    def train(self, num_steps=1000):
        print("Start training...")
        # init optimizers for MatrixA, Reconstructor
        matrix_a_opt = torch.optim.Adam(self.matrix_a.parameters(), lr=0.0002)
        reconstructor_opt = torch.optim.Adam(self.reconstructor.parameters(), lr=0.0002)

        # start training loop
        for step in range(num_steps):
            self.g.zero_grad()
            self.matrix_a.zero_grad()
            self.reconstructor.zero_grad()

            # cast random noise z
            z = make_noise(self.p.batch_size, self.g.dim_z).cuda()
            target_indices, shifts, basis_shift = self.make_shifts(self.matrix_a.input_dim)

            # cast random integer that represents the k^th column  --> e_k
            shift = self.matrix_a(basis_shift)

            # generate images --> from z and from z + A(epsilon * e_k)
            imgs = self.g(z)
            imgs_shifted = self.g.gen_shifted(z, shift) #TODO shift decorator for g

            logits, shift_prediction = self.matrix_a(imgs, imgs_shifted)
            logit_loss = self.label_weight * self.cross_entropy(logits, target_indices)
            shift_loss = self.shift_weight * torch.mean(torch.abs(shift_prediction - shifts))

            # total loss
            loss = logit_loss + shift_loss
            loss.backward()

            matrix_a_opt.step()
            reconstructor_opt.step()

    def gen_animation(self):
        print("generating animation...")

    def generate_noise(self):
        print("generating noise...")
