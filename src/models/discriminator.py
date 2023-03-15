import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_gpu, num_feature_maps, num_color_channels):
        super(Discriminator, self).__init__()
        self.ngpu = num_gpu
        self.main = nn.Sequential(
            nn.Conv2d(num_color_channels, num_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feature_maps, num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feature_maps * 2, num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feature_maps * 4, num_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
