import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(dim, dim),
            nn.Tanh(),

            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):

        weights = self.network(x)

        return x * weights


class SAEA(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim=16
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.attention = Attention(32)

        self.latent = nn.Linear(
            32,
            latent_dim
        )

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, input_dim)
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.attention(x)

        z = self.latent(x)

        return self.decoder(z)

    def encode(self, x):

        x = self.encoder(x)

        x = self.attention(x)

        return self.latent(x)