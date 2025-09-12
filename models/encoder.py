# encodes 512 dimensional CLIP embeddding into a 64 dimensional latent space

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim=512, latent_dim=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, output_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, z):
        return self.net(z)

def rbf_mmd(X, Y, sigma=1.0):
    XX = torch.cdist(X, X)**2
    YY = torch.cdist(Y, Y)**2
    XY = torch.cdist(X, Y)**2

    K_XX = torch.exp(-XX / (2*sigma**2))
    K_YY = torch.exp(-YY / (2*sigma**2))
    K_XY = torch.exp(-XY / (2*sigma**2))

    mmd = K_XX.mean() + K_YY.mean() - 2*K_XY.mean()
    return mmd

def rbf_mmd_efficient(X, Y, sigma=1.0):
    def pdist_sq(X):
        X_norm = (X**2).sum(dim=1).view(-1, 1)
        return X_norm + X_norm.t() - 2*X @ X.t()

    K_XX = torch.exp(-pdist_sq(X)/ (2*sigma**2))
    K_YY = torch.exp(-pdist_sq(Y)/ (2*sigma**2))
    
    XY_norm = (X**2).sum(1).view(-1,1) + (Y**2).sum(1).view(1,-1)
    K_XY = torch.exp(-(XY_norm - 2*X@Y.t()) / (2*sigma**2))
    
    mmd = K_XX.mean() + K_YY.mean() - 2*K_XY.mean()
    return mmd
