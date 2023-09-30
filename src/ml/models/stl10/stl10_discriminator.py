import torch
from torch import nn
import torch.nn.functional as F


class Stl10Discriminator1(nn.Module):
    def __init__(self, num_feature_maps, num_color_channels=3):
        super(Stl10Discriminator1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_color_channels, num_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_feature_maps, num_feature_maps * 2, 4, 3, 2, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_feature_maps * 2, num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(num_feature_maps * 4, 1, 6, 3, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("DISCRIMINATOR")
        # print(x.shape)
        x = self.layer1(x)
#         print(x.shape)
        x = self.layer2(x)
#         print(x.shape)
        x = self.layer3(x)
#         print(x.shape)
        feature = x
        x = self.layer4(x)
#         print(x.shape)
        # return x.view(-1, 1).squeeze(1), feature
#         print("DISCRIMINATOR DONE")
        return x, feature


class Stl10Discriminator2(nn.Module):
    def __init__(self):
        super(Stl10Discriminator2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 0)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 0)

        self.b1 = nn.BatchNorm2d(16)
        self.b2 = nn.BatchNorm2d(32)
        self.b3 = nn.BatchNorm2d(64)
        self.b4 = nn.BatchNorm2d(128)

        self.dense1 = nn.Linear(2 * 2 * 256, 1000)
        self.dense2 = nn.Linear(1000, 500)
        self.dense3 = nn.Linear(500, 70)
        self.dense4 = nn.Linear(70, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.b1(x)

        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.b2(x)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.b3(x)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.b4(x)

        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)

        x = F.leaky_relu(self.dense1(x))
        x = F.dropout(x, 0.5, training=self.training)

        x = F.leaky_relu(self.dense2(x))
        x = F.dropout(x, 0.35, training=self.training)

        x = F.leaky_relu(self.dense3(x))
        x = F.dropout(x, 0.1, training=self.training)

        feature = x
        x = torch.sigmoid(self.dense4(x))

        return x, feature
