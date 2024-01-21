import sys

sys.path.append('ml/models/StyleGAN')

import torch

from src.ml.models.StyleGAN.training.networks import Generator


class StyleGANGeneratorWrapper(Generator):
    def __init__(self, g: Generator):
        super().__init__(g.z_dim,
                         g.c_dim,
                         g.w_dim,
                         g.img_resolution,
                         g.img_channels)

        self.synthesis = g.synthesis
        self.num_ws = g.synthesis.num_ws
        self.mapping = g.mapping

    def forward(self, z, c=None, truncation_psi=0.7, noise_mode='const', **synthesis_kwargs):
        return super(Generator, self).forward(z, c=c, truncation_psi=truncation_psi, noise_mode=noise_mode)

    def gen_shifted(self, x, shift):
        x = torch.reshape(x, (14, 512))
        return self.forward(x + shift, c=None, truncation_psi=0.7, noise_mode='const')
