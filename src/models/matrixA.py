import torch.nn as nn


class MatrixA(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim):
        super(MatrixA, self).__init__()

        self.fc1 = nn.Linear(input_dim, inner_dim)
        self.bn1 = nn.BatchNorm1d(inner_dim)
        self.act1 = nn.ELU()

        self.fc2 = nn.Linear(inner_dim, inner_dim)
        self.bn2 = nn.BatchNorm1d(inner_dim)
        self.act2 = nn.ELU()

        self.fc3 = nn.Linear(inner_dim, inner_dim)
        self.bn3 = nn.BatchNorm1d(inner_dim)
        self.act3 = nn.ELU()

        self.fc4 = nn.Linear(inner_dim, output_dim)

    def forward(self):  # maybe need to init input?
        x1 = self.fc1(input)
        x = self.act1(self.bn1(x1))

        x2 = self.fc2(x)
        x = self.act2(self.bn2(x2 + x1))

        x3 = self.fc3(x)
        x = self.act3(self.bn3(x3 + x2 + x1))

        out = self.fc4(x) + input

        return out
