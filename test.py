import torch
from torch.utils.data import DataLoader
import subprocess
import os
import tempfile
from torchvision.utils import save_image

from models.bayes import NormalBayes
from dataset.embeddings import EmbeddingDataset
from utils import device

def render_image_w3m(image_tensor):
    """Render image tensor using w3m"""
    # Create a temporary file to save the image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    # Save the image tensor to the temporary file
    save_image(image_tensor, temp_path)
    
    # Render with w3m
    subprocess.run(["w3m", temp_path])
    
    # Clean up the temporary file
    os.unlink(temp_path)

def test():
    # Load dataset with metadata support
    dataset = EmbeddingDataset('latent_embeddings', return_metadata=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)

    distribution = NormalBayes()

    mask = torch.ones(len(dataset), dtype=torch.bool).to(device)
    while mask.any():
        # max entropy sampling
        # p(y=1|x,mu,theta)

        # y = 1 means the user likes the image
        # y = 0 means the user dislikes the image
        
        
        # Each batch is (embeddings, metadata). We only need the embeddings tensor.
        for latents, _ in dataloader:
            latents = latents.to(device)
            global_idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            candidates = latents[global_idxs]

            logp = distribution.log_likelihood(candidates, y=1)
            p = torch.exp(logp).clamp(1e-6, 1 - 1e-6)
            bernoulli_entropy = -p*torch.log(p) - (1-p)*torch.log(1-p)

            print('min entropy:', bernoulli_entropy.min().item())
            print('max entropy:', bernoulli_entropy.max().item())
            print('sum entropy:', bernoulli_entropy.sum().item())

            x_local = int(torch.argmax(bernoulli_entropy).item())
            x_global = int(global_idxs[x_local].item())

            ties = (bernoulli_entropy == bernoulli_entropy.max().item()).nonzero(as_tuple=True)[0]
            
            print("argmax:", x_global)
            print("ties:", ties.tolist())
            mask[x_global] = False

            image_tensor = dataset.get_image(x_global)
            prompt = dataset.get_prompt(x_global)
            
            print(f"Prompt: {prompt}")
            render_image_w3m(image_tensor)

            y = input('Do you like this image? (1/0 or "exit" to quit): ')
            if y == 'exit':
                return
            if y != '1' and y != '0':
                print("Please enter 1 or 0")
                continue

            y = int(y)

            # Update distribution with the latent embedding (not the image)
            latent_embedding = latents[x_local]
            distribution.update(latent_embedding, y)

            
if __name__ == '__main__':
    test()


