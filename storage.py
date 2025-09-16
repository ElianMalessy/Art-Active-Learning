import torch
from transformers import CLIPProcessor, CLIPModel
from utils import device
import os
import numpy as np
import pickle
import gc


def store_clip_embeddings(dataloader, save_dir):
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    os.makedirs(save_dir, exist_ok=True)
    all_embeddings = []
    image_metadata = []

    for idx, (images, prompts) in enumerate(dataloader):
        # Preprocess
        inputs = processor(text=prompts, images=images, return_tensors='pt', padding=True, do_rescale=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        all_embeddings.append(image_features.cpu())
        
        # Store metadata for each image in the batch (lightweight mapping to original dataset)
        batch_size = images.shape[0]
        for i in range(batch_size):
            # Store lightweight metadata that allows image retrieval from original dataset
            image_metadata.append({
                'prompt': prompts[i],
                'batch_idx': idx,
                'sample_idx': i,
                'global_idx': len(image_metadata)  # Running global index independent of batch size
            })
        
        if (idx + 1) % 10 == 0:
            print(f'Processed {(idx + 1) * 128} images')
            # Clear cache periodically to prevent memory buildup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy().astype('float32')
    np.save(os.path.join(save_dir, 'embeddings.npy'), all_embeddings, allow_pickle=False)
    
    # Save image metadata
    with open(os.path.join(save_dir, 'image_metadata.pkl'), 'wb') as f:
        pickle.dump(image_metadata, f)

    print(f'All embeddings and metadata saved in {save_dir}')


def store_latent_embeddings(dataloader, encoder, save_dir, source_metadata_path=None):
    os.makedirs(save_dir, exist_ok=True)

    all_embeddings = []
    
    # Load source metadata if provided (from CLIP embeddings)
    source_metadata = None
    if source_metadata_path and os.path.exists(source_metadata_path):
        with open(source_metadata_path, 'rb') as f:
            source_metadata = pickle.load(f)
    
    for idx, clip_embeddings in enumerate(dataloader):
        clip_embeddings = clip_embeddings.to(device)

        with torch.no_grad():
            latents = encoder(clip_embeddings)

        all_embeddings.append(latents.cpu())
        if (idx + 1) % 10 == 0:
            print(f'Processed {(idx + 1) * 128} clip_embeddings')

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy().astype('float32')
    np.save(os.path.join(save_dir, 'embeddings.npy'), all_embeddings, allow_pickle=False)
    
    # Copy metadata from source if available
    if source_metadata:
        with open(os.path.join(save_dir, 'image_metadata.pkl'), 'wb') as f:
            pickle.dump(source_metadata, f)
        print(f'All embeddings and metadata saved in {save_dir}')
    else:
        print(f'All embeddings saved in {save_dir}')

