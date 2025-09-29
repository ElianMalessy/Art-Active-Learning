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
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    save_image(image_tensor, temp_path)
    subprocess.run(["w3m", temp_path])
    os.unlink(temp_path)

def test():
    dataset = EmbeddingDataset('latent_embeddings', return_metadata=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)

    distribution = NormalBayes()

    mask = torch.ones(len(dataset), dtype=torch.bool).to(device)
    print(f"Starting main loop with {mask.sum().item()} candidates")
    iteration = 0
    while mask.any():
        iteration += 1
        
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
            print('p stats:')
            print('min:', p.min().item())
            print('max:', p.max().item())
            
            if iteration <= 10:
                # max entropy sampling for first 10 images
                bernoulli_entropy = -p*torch.log(p) - (1-p)*torch.log(1-p)

                print('min entropy:', bernoulli_entropy.min().item())
                print('max entropy:', bernoulli_entropy.max().item())
                print('sum entropy:', bernoulli_entropy.sum().item())

                x_local = int(torch.argmax(bernoulli_entropy).item())
                
                ties = (torch.where(bernoulli_entropy == bernoulli_entropy.max())[0]).tolist()
                print('ties:', len(ties))
            else:
                # 85% chance of maximum likelihood, 15% chance of max entropy
                if torch.rand(1).item() < 0.85:
                    # maximum likelihood sampling
                    x_local = int(torch.argmax(p).item())
                    print('Using maximum likelihood sampling')
                else:
                    # max entropy sampling
                    bernoulli_entropy = -p*torch.log(p) - (1-p)*torch.log(1-p)
                    x_local = int(torch.argmax(bernoulli_entropy).item())
                    print('Using max entropy sampling')
                    
            x_global = int(global_idxs[x_local].item())
            
            print('argmax:', x_global)
            mask[x_global] = False

            image_tensor = dataset.get_image(x_global)
            prompt = dataset.get_prompt(x_global)
            
            print(f'Prompt: {prompt}')
            render_image_w3m(image_tensor)

            y = input('Do you like this image? (0/1 or "exit" to quit): ')
            if y == 'exit':
                return
            if y != '0' and y != '1':
                print('Please enter 0 or 1')
                continue

            y = int(y)

            # Update distribution with the latent embedding (not the image)
            latent_embedding = latents[x_local]
            distribution.update(latent_embedding, y)

            
if __name__ == '__main__':
    test()


