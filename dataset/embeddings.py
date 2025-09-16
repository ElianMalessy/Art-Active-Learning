import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
from torchvision import transforms
from datasets import load_dataset

class EmbeddingDataset(Dataset):
    def __init__(self, save_dir, return_metadata=False):
        self.embeddings = np.load(os.path.join(save_dir, "embeddings.npy"), mmap_mode="r")
        self.return_metadata = return_metadata
        
        metadata_path = os.path.join(save_dir, "image_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.original_dataset = load_dataset('huggan/wikiart', split='train')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = torch.from_numpy(self.embeddings[idx].copy())
        
        if self.return_metadata and self.metadata:
            return embedding, self.metadata[idx]
        else:
            return embedding
    
    def get_image(self, idx):
        """Get the original image tensor for a given index by retrieving from original dataset"""
        global_idx = self.metadata[idx]['global_idx']
        
        if global_idx >= len(self.original_dataset):
            raise ValueError(f"Global index {global_idx} out of range for original dataset")
        
        item = self.original_dataset[global_idx]
        image = item['image'].convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def get_prompt(self, idx):
        """Get the prompt for a given index"""
        if self.metadata and idx < len(self.metadata):
            return self.metadata[idx]['prompt']
        else:
            raise ValueError(f"No metadata available for index {idx}")
