import torch
from torch import nn


class Stl10Generator(nn.Module):
    def __init__(self, size_z, num_feature_maps, num_color_channels=3):
        super(Stl10Generator, self).__init__()
        self.size_z = size_z

        self.layer1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(size_z, num_feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 8, num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.ReLU(True)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 4, num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.ReLU(True)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 2, num_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True)
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps, num_color_channels, 5, 3, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inp):
        # print("GENERATOR")
        # print(inp.shape)
        out = self.layer1(inp)
#         print(out.shape)
        out = self.layer2(out)
#         print(out.shape)
        out = self.layer3(out)
#         print(out.shape)
        out = self.layer4(out)
#         print(out.shape)
        out = self.layer5(out)
#         print(out.shape)
#         print("GENERATOR DONE")
        return out

    def gen_shifted(self, x, shift):
        shift = torch.unsqueeze(shift, -1)
        shift = torch.unsqueeze(shift, -1)
        return self.forward(x + shift)
