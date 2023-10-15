import torch
from torch import nn


class MvTecGenerator(nn.Module):
    def __init__(self, size_z, num_feature_maps, num_color_channels=3):
        super(MvTecGenerator, self).__init__()
        self.size_z = size_z

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(size_z, num_feature_maps * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_feature_maps * 16),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 16, num_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.ReLU(True)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 8, num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.ReLU(True)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 4, num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.ReLU(True)
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 2, num_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True)
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps, num_color_channels, 4, 2, 1, bias=False),
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
        out = self.layer6(out)
#         print(out.shape)
#         print("GENERATOR DONE")
        return out

    def gen_shifted(self, x, shift):
        shift = torch.unsqueeze(shift, -1)
        shift = torch.unsqueeze(shift, -1)
        return self.forward(x + shift)
