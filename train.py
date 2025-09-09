import faiss


def train():
    cpu_index = faiss.read_index("wikiart_index.faiss")
    while True:
        # max entropy
        for i in range(cpu_index.ntotal):
            emb = cpu_index.reconstruct(i)
            


