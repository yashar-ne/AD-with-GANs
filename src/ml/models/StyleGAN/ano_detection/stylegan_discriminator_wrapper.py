import sys

sys.path.append('ml/models/StyleGAN')

from src.ml.models.StyleGAN.training.networks import Discriminator


class StyleGANDiscriminatorWrapper(Discriminator):
    def __init__(self, d: Discriminator):
        super().__init__(
            d.c_dim,
            d.img_resolution,
            d.img_channels,
        )

        self.img_resolution_log2 = d.img_resolution_log2
        self.block_resolutions = d.block_resolutions
        if d.mapping is not None:
            self.mapping = d.mapping
        self.b4 = d.b4

    def forward(self, z, c=None, **block_kwargs):
        return super(Discriminator, self).forward(z, c=c, block_kwargs=block_kwargs)
