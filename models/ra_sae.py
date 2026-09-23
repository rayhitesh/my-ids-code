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

        residual = x

        output = self.layers(x)

        return self.activation(
            residual + output
        )


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


class RA_SAE(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim=16
    ):

        super().__init__()

        # Encoder

        self.input_layer = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.ReLU()
        )

        self.residual = \
            ResidualBlock(64)

        self.encoder = nn.Sequential(

            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Attention

        self.attention = Attention(32)

        # Latent representation

        self.latent = nn.Linear(
            32,
            latent_dim
        )

        # Decoder

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.decoder_residual = \
            ResidualBlock(64)

        self.output = nn.Linear(
            64,
            input_dim
        )

    def forward(self, x):

        x = self.input_layer(x)

        x = self.residual(x)

        x = self.encoder(x)

        x = self.attention(x)

        z = self.latent(x)

        x = self.decoder(z)

        x = self.decoder_residual(x)

        return self.output(x)

    def encode(self, x):

        x = self.input_layer(x)

        x = self.residual(x)

        x = self.encoder(x)

        x = self.attention(x)

        return self.latent(x)