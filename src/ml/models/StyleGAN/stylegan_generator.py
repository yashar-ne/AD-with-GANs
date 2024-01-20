import sys

sys.path.append('ml/models/StyleGAN')

import torch

from src.ml.models.StyleGAN.training.networks import Generator


class StyleGANGenerator(Generator):
    def gen_shifted(self, x, shift):
        shift = torch.unsqueeze(shift, -1)
        shift = torch.unsqueeze(shift, -1)
        return self.forward(x + shift)
