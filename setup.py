import torch
from transformers import CLIPProcessor, CLIPModel
from models import device
import os

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

def store_embeddings(dataset, save_dir="wikiart_embeddings"):
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, (images, prompts) in enumerate(dataset):
        # Preprocess
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Save this batch to disk
        save_path = os.path.join(save_dir, f"batch_{batch_idx}.pt")
        torch.save({'embeddings': image_features.cpu(), 'prompts': prompts}, save_path)

        if batch_idx % 10 == 0:
            print(f"Saved {batch_idx+1} batches so far...")

    print(f"All embeddings saved in {save_dir}")
