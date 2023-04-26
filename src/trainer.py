import torch

from src.models.generator import Generator
from src.models.matrix_a_linear import MatrixALinear
from src.models.reconstructor import Reconstructor


class Trainer:
    def __init__(self, z_dim, directions_count, max_latent_dim, epsilon):
        super(Trainer, self).__init__()
        # init Generator
        self.g = Generator(size_z=z_dim, num_feature_maps=64, num_color_channels=1)

        # init MatrixA
        self.matrix_a = MatrixALinear(input_dim=directions_count, inner_dim=1024, output_dim=max_latent_dim)

        # init Reconstructor
        self.reconstructor = Reconstructor(dim=self.matrix_a.input_dim)

    def load_models_from_checkpoints(self):
        print("loading models...")

    def train(self):
        print("Start training...")
        # init optimizers for MatrixA, Reconstructor
        matrix_a_opt = torch.optim.Adam(self.matrix_a.parameters(), lr=0.0002)
        reconstructor_opt = torch.optim.Adam(self.reconstructor.parameters(), lr=0.0002)

        # start training loop
        # cast random integer that represents the k^th column  --> e_k


        # calculate shift

        # cast random noise z

        # calculate shifted z --> z + A(epsilon * e_k)

        # generate images --> from z and from z + A(epsilon * e_k)

        # calculate loss

        # optimizers step

    def gen_animation(self):
        print("generating animation...")

    def generate_noise(self):
        print("generating noise...")
