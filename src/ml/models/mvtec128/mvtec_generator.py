import torch
from torch import nn


class MvTecGenerator(nn.Module):
    def __init__(self, size_z, num_feature_maps, num_color_channels=3, dropout_rate=0):
        super(MvTecGenerator, self).__init__()
        self.size_z = size_z
        self.dropout_rate = dropout_rate

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(size_z, num_feature_maps * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_feature_maps * 16),
            nn.ReLU(True)
        )

        self.dropout1 = nn.Dropout2d(self.dropout_rate)

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 16, num_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.ReLU(True)
        )

        self.dropout2 = nn.Dropout2d(self.dropout_rate)

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 8, num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.ReLU(True)
        )

        self.dropout3 = nn.Dropout2d(self.dropout_rate)

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 4, num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.ReLU(True)
        )

        self.dropout4 = nn.Dropout2d(self.dropout_rate)

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 2, num_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True)
        )

        self.dropout5 = nn.Dropout2d(self.dropout_rate)

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps, num_color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inp):
        # print("GENERATOR")
        # print(inp.shape)
        out = self.layer1(inp)
        out = self.dropout1(out)
#         print(out.shape)
        out = self.layer2(out)
        out = self.dropout2(out)
#         print(out.shape)
        out = self.layer3(out)
        out = self.dropout3(out)
#         print(out.shape)
        out = self.layer4(out)
        out = self.dropout4(out)
#         print(out.shape)
        out = self.layer5(out)
        out = self.dropout5(out)
#         print(out.shape)
        out = self.layer6(out)
#         print(out.shape)
#         print("GENERATOR DONE")
        return out

    def gen_shifted(self, x, shift):
        shift = torch.unsqueeze(shift, -1)
        shift = torch.unsqueeze(shift, -1)
        return self.forward(x + shift)
