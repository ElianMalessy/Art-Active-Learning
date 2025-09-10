import faiss
import torch
from torch.utils.data import DataLoader

from dataset.embeddings import EmbeddingDataset
from models.encoder import Encoder
from models import device


def train():
    index = faiss.read_index("wikiart_index.faiss")
    d = 64
    latent_index = faiss.IndexFlatIP(d)

    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()  # initialize GPU resources
        index = faiss.index_cpu_to_gpu(res, 0, index)
        latent_index = faiss.index_cpu_to_gpu(res, 0, latent_index)


    dataset = EmbeddingDataset("wikiart_embeddings")
    batch_size = 512
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    encoder = Encoder().to(device)
    for embeddings, _ in dataloader:
        embeddings = embeddings.to(device)
        latents = encoder(embeddings)

        # loss = ...

            
if __name__ == '__main__':
    train()


