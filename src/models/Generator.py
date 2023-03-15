import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_gpu, num_feature_maps, num_color_channels):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(num_color_channels, num_feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feature_maps * 8, num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feature_maps * 4, num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feature_maps * 2, num_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feature_maps, num_color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)