import torch
from torch import nn


class CelebGenerator(nn.Module):
    def __init__(self, size_z, num_feature_maps, num_color_channels=3):
        super(CelebGenerator, self).__init__()
        self.size_z = size_z
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(size_z, num_feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(num_feature_maps * 8, num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(num_feature_maps * 4, num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(num_feature_maps * 2, num_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(num_feature_maps, num_color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

    def gen_shifted(self, x, shift):
        shift = torch.unsqueeze(shift, -1)
        shift = torch.unsqueeze(shift, -1)
        return self.forward(x + shift)


class CelebGenerator128(nn.Module):
    def __init__(self, size_z, num_feature_maps, num_color_channels=3):
        super(CelebGenerator128, self).__init__()
        self.size_z = size_z
        self.main = nn.Sequential(  # input is Z, going into a convolution
            nn.ConvTranspose2d(size_z, num_feature_maps * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_feature_maps * 16),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(num_feature_maps * 16, num_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(num_feature_maps * 8, num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(num_feature_maps * 4, num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(num_feature_maps * 2, num_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True),
            # state size. ``(ngf) x 64 x 64``
            nn.ConvTranspose2d(num_feature_maps, num_color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 128 x 128``
        )

    def forward(self, input):
        return self.main(input)

    def gen_shifted(self, x, shift):
        shift = torch.unsqueeze(shift, -1)
        shift = torch.unsqueeze(shift, -1)
        return self.forward(x + shift)
