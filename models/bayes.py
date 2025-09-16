import torch
from utils import device

class NormalBayes():
    def __init__(self, input_dim=64, classes=(0,1)):
        self.input_dim = input_dim
        self.classes = classes
        self.device = device

        self.counts = {}

        self.mu = {}
        self.kappa = {}
        self.nu = {}
        self.psi = {}

        for y in self.classes:
            self.counts[y] = 0

            self.mu[y] = torch.zeros(self.input_dim, device=device)
            self.kappa[y] = 1.0
            self.nu[y] = float(self.input_dim + 2)
            self.psi[y] = torch.eye(self.input_dim, device=device)

    def update(self, x, y):
        self.counts[y] += 1
        x = x.unsqueeze(0)

        n = x.shape[0]
        x_bar = x.mean(0)
        Xc = x - x_bar
        S = Xc.T @ Xc

        mu0 = self.mu[y]
        kappa0 = self.kappa[y]
        nu0 = self.nu[y]
        psi0 = self.psi[y]

        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
        nu_n = nu0 + n
        diff = (x_bar - mu0).unsqueeze(1)
        psi_n = psi0 + S + (kappa0 * n / kappa_n) * (diff @ diff.T)

        self.mu[y] = mu_n
        self.kappa[y] = kappa_n
        self.nu[y] = nu_n
        self.psi[y] = psi_n

    def class_prior(self, y):
        total = sum(self.counts.values())
        return (self.counts[y] + 1.0) / (total + len(self.classes))

    def _studentt_logpdf(self, x, y):
        x = x.unsqueeze(0)
        x = x.squeeze(0)
        D = self.input_dim

        mu_n = self.mu[y]
        kappa_n = self.kappa[y]
        nu_n = self.nu[y]
        psi_n = self.psi[y]

        nu_t = nu_n - D + 1
        scale = (kappa_n + 1) / (kappa_n * nu_t) * psi_n

        diff = (x - mu_n).unsqueeze(1)
        sol = torch.linalg.solve(scale, diff)
        maha = (diff.t() @ sol).squeeze()

        _, logdet = torch.linalg.slogdet(scale)

        term1 = torch.lgamma(torch.tensor((nu_t + D) / 2.0, device=self.device))
        term2 = torch.lgamma(torch.tensor(nu_t / 2.0, device=self.device))
        term3 = 0.5 * (logdet + D * torch.log(torch.tensor(nu_t * torch.pi, device=self.device)))
        term4 = ((nu_t + D) / 2.0) * torch.log1p(maha / nu_t)

        return term1 - term2 - term3 - term4

    def log_likelihood(self, x, y):
        log_pxy = {}
        for c in self.classes:
            log_px_given_y = self._studentt_logpdf(x, c)
            log_prior = torch.log(torch.tensor(self.class_prior(c), device=self.device))
            log_pxy[c] = log_px_given_y + log_prior

        stack = torch.stack([log_pxy[c] for c in self.classes])
        log_px = torch.logsumexp(stack, dim=0)
        return log_pxy[y] - log_px
