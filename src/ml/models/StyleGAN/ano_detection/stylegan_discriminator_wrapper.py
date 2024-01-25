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

        self.c_dim = d.c_dim
        self.img_resolution = d.img_resolution
        self.img_resolution_log2 = d.img_resolution_log2
        self.img_channels = d.img_channels
        self.block_resolutions = d.block_resolutions
        if hasattr(d, "mapping"):
            self.mapping = d.mapping
        self.b4 = d.b4

    def forward(self, img, c=None, **block_kwargs):
        return super(Discriminator, self).forward(img, c=c, block_kwargs=block_kwargs)
