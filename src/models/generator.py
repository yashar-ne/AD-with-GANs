import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_gpu, size_z, num_feature_maps, num_color_channels):
        super(Generator, self).__init__()
        self.ngpu = num_gpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(size_z, num_feature_maps, 3, 1, 0, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feature_maps, num_feature_maps, 3, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feature_maps, num_feature_maps, 3, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feature_maps, num_feature_maps, 3, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feature_maps, num_color_channels, 3, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)