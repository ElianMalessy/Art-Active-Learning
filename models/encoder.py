# encodes 512 dimensional CLIP embeddding into a 64 dimensional latent space

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=512, latent_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim)  # enforces zero-mean, unit variance per batch
        )

    def forward(self, x):
        return self.mlp(x)
