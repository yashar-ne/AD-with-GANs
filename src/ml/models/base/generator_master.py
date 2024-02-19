import torch
import torch.nn as nn


class GeneratorMaster(nn.Module):
    def __init__(self, z_dim, num_feature_maps, num_channels, dropout_rate=0):
        super(GeneratorMaster, self).__init__()
        self.z_dim = z_dim
        self.num_feature_maps = num_feature_maps
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.network = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, num_feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.ReLU(True),

            nn.Dropout2d(self.dropout_rate),

            nn.ConvTranspose2d(num_feature_maps * 8, num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.ReLU(True),

            nn.Dropout2d(self.dropout_rate),

            nn.ConvTranspose2d(num_feature_maps * 4, num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.ReLU(True),

            nn.Dropout2d(self.dropout_rate),

            nn.ConvTranspose2d(num_feature_maps * 2, num_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_feature_maps, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.network(x)
        return output

    def gen_shifted(self, x, shift):
        shift = torch.unsqueeze(shift, -1)
        shift = torch.unsqueeze(shift, -1)
        return self.forward(x + shift)
