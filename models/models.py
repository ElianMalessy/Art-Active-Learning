import torch
from models import device

class NormalBayes():
    def __init__(self, input_dim=512, sigma=1):
        super(NormalBayes, self).__init__()
        self.input_dim = input_dim
        self.sigma = sigma
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
            self.dist[y] = torch.distributions.MultivariateNormal(self.mu[y], self.cov[y])


    def update(self, x, y):
        self.counts[y] += 1

        inv_prior_cov = torch.linalg.inv(self.cov[y])
        posterior_cov = torch.linalg.inv(inv_prior_cov + self.inv_obs_cov[y])
        self.mu[y] = posterior_cov @ ((inv_prior_cov @ self.mu[y]) + (self.inv_obs_cov @ x))
        self.cov[y] = posterior_cov

        self.dist[y] = torch.distributions.MultivariateNormal(self.mu[y], self.cov[y])

    # p(y=1)
    def class_prior(self, y):
        total = sum(self.counts.values())
        # Laplace smoothing to avoid zero
        return (self.counts[y] + 1.0) / (total + len(self.classes))

    # p(y=1|x) = p(x|y=1)p(y=1)/(p(x|y=1)p(y=1) + p(x|y=0)p(y=0))
    def log_probability(self, x, y):
        log_pxy = {}
        for c in self.classes:
            log_pxy[c] = self.dist[c].log_prob(x) + torch.log(torch.tensor(self.class_prior(c), device=self.device))

        # Log-sum-exp for normalization
        log_px = torch.logsumexp(torch.stack([log_pxy[0], log_pxy[1]]), dim=0)
        log_pyx = log_pxy[y] - log_px
        return log_pyx
        




