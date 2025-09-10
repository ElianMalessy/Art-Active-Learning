import faiss
import torch
from models.bayes import NormalBayes


def test():
    index = faiss.read_index("embedding_index.faiss")

    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()  # initialize GPU resources
        index = faiss.index_cpu_to_gpu(res, 0, index)


    distribution = NormalBayes()
    embeddings = index.reconstruct_n(0, index.ntotal)

    while True:
        # max entropy sampling
        # p(y=1|x,mu,theta)

        logp = distribution.log_likelihood(embeddings, y=1)
        p = torch.exp(logp)
        bernoulli_entropy = -p*logp - (1-p)*torch.log(1-p)

        x = torch.argmax(bernoulli_entropy)
        y = input()
        distribution.update(x, y)

            
if __name__ == '__main__':
    test()


