import torch
from utils import device

class NormalBayes():
    def __init__(self, input_dim=64, sigma=1, jitter=1e-3):
        self.input_dim = input_dim
        self.sigma = sigma
        self.jitter = jitter
        self.device = device

        self.classes = [0, 1]
        self.mu = {}
        self.cov = {}
        self.inv_obs_cov = (1.0 / self.sigma**2) * torch.eye(self.input_dim, device=device)
        self.dist = {}
        self.counts = {0: 0, 1: 0}

        for y in self.classes:
            self.mu[y] = torch.zeros(self.input_dim, device=device)
            self.cov[y] = torch.eye(self.input_dim, device=device)
            self._rebuild_dist(y)

    def _make_spd(self, cov):
        # Symmetrize
        cov = 0.5 * (cov + cov.T)
        # Ensure positive definiteness by shifting eigenvalues if needed
        eps = self.jitter
        try:
            min_eig = torch.linalg.eigvalsh(cov).min().real
        except RuntimeError:
            min_eig = torch.tensor(-1.0, device=self.device)
        shift = torch.clamp(eps - min_eig, min=0.0)
        if shift > 0:
            cov = cov + (shift + eps) * torch.eye(self.input_dim, device=self.device)
        else:
            cov = cov + eps * torch.eye(self.input_dim, device=self.device)
        return cov

    def _rebuild_dist(self, y):
        cov_spd = self._make_spd(self.cov[y])
        self.dist[y] = torch.distributions.MultivariateNormal(self.mu[y], cov_spd)


    def update(self, x, y):
        self.counts[y] += 1

        inv_prior_cov = torch.linalg.inv(self._make_spd(self.cov[y]))
        # Use Cholesky-based inverse for stability
        A = inv_prior_cov + self.inv_obs_cov
        try:
            L = torch.linalg.cholesky(self._make_spd(A))
            posterior_cov = torch.cholesky_inverse(L)
        except RuntimeError:
            posterior_cov = torch.linalg.inv(self._make_spd(A))
        self.mu[y] = posterior_cov @ ((inv_prior_cov @ self.mu[y]) + (self.inv_obs_cov @ x))
        self.cov[y] = posterior_cov

        # Recreate distribution with SPD covariance
        self._rebuild_dist(y)

    # p(y=1)
    def class_prior(self, y):
        total = sum(self.counts.values())
        # Laplace smoothing to avoid zero
        return (self.counts[y] + 1.0) / (total + len(self.classes))

    # p(y=1|x) = p(x|y=1)p(y=1)/(p(x|y=1)p(y=1) + p(x|y=0)p(y=0))
    def log_likelihood(self, x, y):
        log_pxy = {}
        for c in self.classes:
            lp = self.dist[c].log_prob(x)
            # Replace potential NaNs from upstream with large negative
            lp = torch.nan_to_num(lp, nan=-1e30, posinf=1e30, neginf=-1e30)
            log_pxy[c] = lp + torch.log(torch.tensor(self.class_prior(c), device=self.device))

        # Log-sum-exp for normalization
        log_px = torch.logsumexp(torch.stack([log_pxy[0], log_pxy[1]]), dim=0)
        log_pyx = log_pxy[y] - log_px
        return log_pyx
        




