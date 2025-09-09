import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NormalBayes():
    def __init__(self, input_dim=512, sigma=1):
        super(NormalBayes, self).__init__()
        self.mu = torch.randn(512, device=device)
        self.covariance = torch.eye(512, device=device)
        self.inv_obs_covariance = torch.linalg.inv((sigma**2) * torch.eye(input_dim))

    def update(self, x):
        inv_prior_covariance = torch.linalg.inv(self.covariance)
        posterior_covariance = torch.linalg.inv(inv_prior_covariance + self.inv_obs_covariance)
        self.mu = torch.matmul(posterior_covariance, (torch.matmul(inv_prior_covariance, self.mu) + torch.matmul(self.inv_obs_covariance, x)))
        self.covariance = posterior_covariance

