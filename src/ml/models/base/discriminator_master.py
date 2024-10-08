import torch.nn as nn


class DiscriminatorMaster(nn.Module):
    def __init__(self, num_feature_maps, num_channels, dropout_rate=0):
        super(DiscriminatorMaster, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_feature_maps = num_feature_maps
        self.num_color_channels = num_channels

        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, num_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.dropout1 = nn.Dropout2d(self.dropout_rate)

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_feature_maps, num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.dropout2 = nn.Dropout2d(self.dropout_rate)

        self.layer3 = nn.Sequential(
            nn.Conv2d(num_feature_maps * 2, num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.dropout3 = nn.Dropout2d(self.dropout_rate)

        self.layer4 = nn.Sequential(
            nn.Conv2d(num_feature_maps * 4, num_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.dropout4 = nn.Dropout2d(self.dropout_rate)

        self.layer5 = nn.Sequential(
            nn.Conv2d(num_feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.dropout1(out)
        out = self.layer2(out)
        out = self.dropout2(out)
        out = self.layer3(out)
        out = self.dropout3(out)
        out = self.layer4(out)
        out = self.dropout4(out)
        feature = out
        out = self.layer5(out)
        return out.view(-1, 1).squeeze(1), feature
