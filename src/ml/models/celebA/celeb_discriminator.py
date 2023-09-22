from torch import nn


class CelebDiscriminator(nn.Module):
    def __init__(self, num_feature_maps, num_color_channels=3):
        super(CelebDiscriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_color_channels, num_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_feature_maps, num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_feature_maps * 2, num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(num_feature_maps * 4, num_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Conv2d(num_feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        feature = out
        out = self.fc(out)
        # return out.view(-1, 1).squeeze(1), feature
        return out, feature


class CelebDiscriminator128(nn.Module):
    def __init__(self, num_feature_maps, num_color_channels=3):
        super(CelebDiscriminator128, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_color_channels, num_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_feature_maps, num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_feature_maps * 2, num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(num_feature_maps * 4, num_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(num_feature_maps * 8, num_feature_maps * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Conv2d(num_feature_maps * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        feature = out
        out = self.fc(out)
        # return out.view(-1, 1).squeeze(1), feature
        return out, feature