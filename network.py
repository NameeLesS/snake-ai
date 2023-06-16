from torch import nn


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.Flatten(),

            nn.Linear(in_features=3136, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=4),
        )

    def forward(self, x):
        x = self.model(x)
        return x
