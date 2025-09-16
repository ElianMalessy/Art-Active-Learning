import torch
import torch.nn as nn

from utils import latent_dim
import math

# encodes 512 dimensional CLIP embeddding into a latent space
class Encoder(nn.Module):
    def __init__(self, input_dim=512, latent_dim=latent_dim, hidden=256):
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
    def __init__(self, latent_dim=latent_dim, output_dim=512, hidden=256):
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

import math
from typing import Iterable, Optional, Tuple

import torch


def pairwise_sq_dists(X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute matrix of squared pairwise distances between rows of X and rows of Y.
    If Y is None, compute pairwise distances between rows of X.
    Returns a (n, m) tensor of squared distances.
    """
    if Y is None:
        Y = X
    # ||x||^2 shape (n,1), ||y||^2 shape (1,m)
    x_norm = X.pow(2).sum(dim=1, keepdim=True)
    y_norm = Y.pow(2).sum(dim=1, keepdim=True).t()
    d2 = x_norm + y_norm - 2.0 * (X @ Y.t())
    # numerical safety: clamp tiny negative values to zero
    return torch.clamp(d2, min=0.0)


def median_heuristic_sigmas(
    z: torch.Tensor,
    factors: Iterable[float] = (0.5, 1.0, 2.0, 4.0),
    subsample: int = 1000,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute a set of RBF bandwidths (sigmas) using the median heuristic.
    Returns a 1-D torch.Tensor of sigmas on the same device/dtype as `z`.
    - factors: multiplies of the base sigma to return multiple bandwidths.
    - subsample: if n > subsample, compute median on a random subset for speed.
    """
    n = z.size(0)
    if n == 0:
        raise ValueError("z must contain at least one sample")
    if n > subsample:
        idx = torch.randperm(n, device=z.device)[:subsample]
        z_sub = z[idx]
    else:
        z_sub = z
    d2 = pairwise_sq_dists(z_sub, z_sub)
    # upper-triangular off-diagonal entries
    if d2.shape[0] < 2:
        base = 1.0
    else:
        tri = torch.triu_indices(d2.shape[0], d2.shape[0], offset=1)
        vals = d2[tri[0], tri[1]]
        if vals.numel() == 0:
            base = 1.0
        else:
            med = float(vals.median().cpu().item())
            base = math.sqrt(max(med, eps))
    sigmas = torch.tensor([base * float(f) for f in factors], device=z.device, dtype=z.dtype)
    return sigmas

from typing import Iterable, Union, Optional
def rbf_multi_sigma(Ksq: torch.Tensor, sigmas: Union[torch.Tensor, Iterable[float]], eps: float = 1e-12) -> torch.Tensor:
    """
    Compute sum_k exp(-||x-y||^2 / (2*sigma_k^2)) given squared-distance matrix Ksq.
    Ksq: (n,m) squared distances tensor.
    sigmas: iterable of scalars or 1-D tensor of sigmas.
    Returns kernel matrix (n,m).
    """
    K = torch.zeros_like(Ksq)
    # allow sigmas to be tensor or list
    for s in sigmas:
        s_val = float(s)  # ensures compatibility if s is torch scalar
        gamma = 1.0 / (2.0 * (s_val ** 2) + eps)
        K = K + torch.exp(-gamma * Ksq)
    return K


def mmd_unbiased_multi_sigma(x: torch.Tensor, y: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Unbiased estimate of MMD^2 between samples x and y using multi-bandwidth RBF kernels.
    Returns a scalar torch.Tensor >= 0.

    Notes:
    - Uses unbiased sums for Kxx and Kyy (diagonals excluded).
    - If sigmas is None, compute median-heuristic sigmas from x.
    """
    n = x.size(0)
    m = y.size(0)
    if n < 2 or m < 2:
        raise ValueError("Need at least 2 samples in each sample set for unbiased MMD")
    if sigmas is None:
        sigmas = median_heuristic_sigmas(x)

    Kxx = rbf_multi_sigma(pairwise_sq_dists(x, x), sigmas)
    Kyy = rbf_multi_sigma(pairwise_sq_dists(y, y), sigmas)
    Kxy = rbf_multi_sigma(pairwise_sq_dists(x, y), sigmas)

    sum_Kxx = Kxx.sum() - Kxx.diag().sum()     # exclude diagonal
    sum_Kyy = Kyy.sum() - Kyy.diag().sum()     # exclude diagonal
    mmd2 = (sum_Kxx / (n * (n - 1))) + (sum_Kyy / (m * (m - 1))) - 2.0 * Kxy.mean()
    # numeric safety
    return torch.clamp(mmd2, min=0.0)
