import torch
from torch.utils.data import DataLoader
import subprocess
import os
import tempfile
from torchvision.utils import save_image

from models.bradley_terry import BradleyTerryModel
from dataset.embeddings import EmbeddingDataset
from utils import device

def render_images_side_by_side(image_tensor_a, image_tensor_b):
    """Display two images side by side using w3m"""
    with tempfile.NamedTemporaryFile(suffix='_a.png', delete=False) as tmp_file_a:
        temp_path_a = tmp_file_a.name
    with tempfile.NamedTemporaryFile(suffix='_b.png', delete=False) as tmp_file_b:
        temp_path_b = tmp_file_b.name
    
    save_image(image_tensor_a, temp_path_a)
    save_image(image_tensor_b, temp_path_b)
    
    # Display both images
    subprocess.run(["w3m", temp_path_a])
    print("--- VS ---")
    subprocess.run(["w3m", temp_path_b])
    
    os.unlink(temp_path_a)
    os.unlink(temp_path_b)

def test():
    dataset = EmbeddingDataset('latent_embeddings', return_metadata=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)

    model = BradleyTerryModel(input_dim=64)

    mask = torch.ones(len(dataset), dtype=torch.bool).to(device)
    print(f"Starting Bradley-Terry comparison learning with {mask.sum().item()} candidates")
    
    iteration = 0
    while mask.sum() >= 2:  # Need at least 2 images for comparison
        iteration += 1
        
        # Get all latent embeddings
        for latents, _ in dataloader:
            latents = latents.to(device)
            global_idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            candidates = latents[global_idxs]

            if len(candidates) < 2:
                print("Not enough candidates remaining for comparison")
                break

            # Select most informative pair using Bradley-Terry model
            local_idx_a, local_idx_b = model.select_most_informative_pair(candidates)
            
            if local_idx_a is None or local_idx_b is None:
                print("Could not find a good pair to compare")
                break
                
            global_idx_a = int(global_idxs[local_idx_a].item())
            global_idx_b = int(global_idxs[local_idx_b].item())
            
            # Remove both from consideration
            mask[global_idx_a] = False
            mask[global_idx_b] = False

            # Get images and prompts
            image_tensor_a = dataset.get_image(global_idx_a)
            image_tensor_b = dataset.get_image(global_idx_b)
            prompt_a = dataset.get_prompt(global_idx_a)
            prompt_b = dataset.get_prompt(global_idx_b)
            
            print(f"\n=== Comparison {iteration} ===")
            print(f'IMAGE A: {prompt_a}')
            print(f'IMAGE B: {prompt_b}')
            
            render_images_side_by_side(image_tensor_a, image_tensor_b)

            choice = input('Which do you prefer? (a/b or "exit" to quit): ').lower().strip()
            if choice == 'exit':
                return
            if choice not in ['a', 'b']:
                print('Please enter "a" or "b"')
                continue

            # Update Bradley-Terry model with comparison result
            embedding_a = latents[local_idx_a]
            embedding_b = latents[local_idx_b]
            a_wins = (choice == 'a')
            
            loss = model.update(embedding_a, embedding_b, a_wins)
            print(f'Model updated. Training loss: {loss:.4f}')
            print(f'Total comparisons made: {model.comparisons_seen}')
            
            # Show current quality predictions for these images
            with torch.no_grad():
                quality_a = model.quality_score(embedding_a).item()
                quality_b = model.quality_score(embedding_b).item()
                print(f'Quality scores: A={quality_a:.3f}, B={quality_b:.3f}')
            
            # Every 5 comparisons, show top recommendations
            if iteration % 5 == 0:
                print(f"\n🎯 TOP 5 RECOMMENDATIONS after {iteration} comparisons:")
                all_embeddings = latents
                top_indices, top_scores = model.get_top_recommendations(all_embeddings)
                
                for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
                    prompt = dataset.get_prompt(idx)
                    print(f"{rank}. [{score:.3f}] {prompt[:80]}...")
                print()
            
            break  # Break the dataloader loop and continue to next iteration

            
if __name__ == '__main__':
    test()


