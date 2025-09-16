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
    
        mu0 = self.mu[y]
        kappa0 = self.kappa[y]
        nu0 = self.nu[y]
        psi0 = self.psi[y]
    
        n = 1
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + x) / kappa_n
        nu_n = nu0 + n
        diff = (x - mu0).unsqueeze(1)
        psi_n = psi0 + (kappa0 * n / kappa_n) * (diff @ diff.T)
    
        self.mu[y] = mu_n
        self.kappa[y] = kappa_n
        self.nu[y] = nu_n
        self.psi[y] = psi_n

    def class_prior(self, y):
        total = sum(self.counts.values())
        return (self.counts[y] + 1.0) / (total + len(self.classes))

    def _studentt_logpdf(self, x, y):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        mu_n = self.mu[y]
        kappa_n = self.kappa[y]
        nu_n = self.nu[y]
        psi_n = self.psi[y]

        nu_t = nu_n - self.input_dim + 1
        scale = (kappa_n + 1) / (kappa_n * nu_t) * psi_n

        # Add regularization to prevent singular matrices
        regularization = 1e-6 * torch.eye(self.input_dim, device=self.device)
        scale = scale + regularization

        diff = x - mu_n
        sol = torch.linalg.solve(scale, diff.T)
        maha = torch.sum(diff.T * sol, dim=0)
        
        # Mahalanobis distance should be non-negative
        maha = torch.clamp(maha, min=0.0)

        _, logdet = torch.linalg.slogdet(scale)

        term1 = torch.lgamma(torch.tensor((nu_t + self.input_dim) / 2.0, device=self.device))
        term2 = torch.lgamma(torch.tensor(nu_t / 2.0, device=self.device))
        term3 = 0.5 * (logdet + self.input_dim * torch.log(torch.tensor(nu_t * torch.pi, device=self.device)))
        
        # add numerical stability for term4
        maha_ratio = maha / nu_t
        # clamp to prevent overflow in log1p
        maha_ratio = torch.clamp(maha_ratio, max=1e6)
        term4 = ((nu_t + self.input_dim) / 2.0) * torch.log1p(maha_ratio)

        result = term1 - term2 - term3 - term4
        
        return result

    def log_likelihood(self, x, y):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        log_pxy = {}
        for c in self.classes:
            log_px_given_y = self._studentt_logpdf(x, c)
            log_prior = torch.log(torch.tensor(self.class_prior(c), device=self.device))
            log_pxy[c] = log_px_given_y + log_prior

        stack = torch.stack([log_pxy[c] for c in self.classes])
        log_px = torch.logsumexp(stack, dim=0)
        return log_pxy[y] - log_px
