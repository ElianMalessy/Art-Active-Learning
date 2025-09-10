import faiss
import numpy as np
import torch
from models.models import NormalBayes


def train():
    index = faiss.read_index("wikiart_index.faiss")

    # if torch.cuda.is_available():
        # res = faiss.StandardGpuResources()  # initialize GPU resources
        # index = faiss.index_cpu_to_gpu(res, 0, index)

    n, dim = index.ntotal, index.d

    distribution = NormalBayes()
    embeddings = index.reconstruct_n(0, n)
    while True:
        # max entropy

        # p(y=1|x,mu,theta)
        logp = distribution.log_likelihood(embeddings, y=1)
        p = torch.exp(logp)
        bernoulli_entropy = -p*logp - (1-p)*torch.log(1-p)

        x = torch.argmax(bernoulli_entropy)
        y = input()
        distribution.update(x, y)

            
            


