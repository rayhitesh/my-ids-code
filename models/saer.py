import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(dim, dim),
            nn.ReLU(),

            nn.Linear(dim, dim)
        )

        self.activation = nn.ReLU()

    def forward(self, x):

        return self.activation(
            x + self.layers(x)
        )


class SAER(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim=16
    ):

        super().__init__()

        self.input_layer = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.ReLU()
        )

        self.residual_encoder = \
            ResidualBlock(64)

        self.encoder = nn.Sequential(

            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.latent = nn.Linear(
            32,
            latent_dim
        )

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.residual_decoder = \
            ResidualBlock(64)

        self.output = nn.Linear(
            64,
            input_dim
        )

    def forward(self, x):

        x = self.input_layer(x)

        x = self.residual_encoder(x)

        x = self.encoder(x)

        z = self.latent(x)

        x = self.decoder(z)

        x = self.residual_decoder(x)

        return self.output(x)

    def encode(self, x):

        x = self.input_layer(x)

        x = self.residual_encoder(x)

        x = self.encoder(x)

        return self.latent(x)