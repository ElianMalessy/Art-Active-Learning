import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 64

images_embeddings = pd.DataFrame(columns=['idx', 'CLIP', 'latents'])
