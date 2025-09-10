import numpy as np
from transformers import CLIPProcessor, CLIPModel
import faiss
import torch
from models import device

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

def store_embeddings(dataset):
    print("Storing embeddings...")

    # CPU index only
    d = 512  # CLIP ViT-B/32 image embeddings have 512 dims
    index = faiss.IndexFlatIP(d)
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    for images, prompts in dataset:
        # Preprocess and move images to GPU for model
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get image embeddings
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Convert to numpy and add to CPU FAISS index
        emb = image_features.detach().cpu().numpy().astype(np.float32)
        index.add(emb)

        if index.ntotal % 100 == 0:
            print(f"Stored {index.ntotal} embeddings so far...")

    # Save index to disk
    if torch.cuda.is_available():
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, "wikiart_index.faiss")

    print("Index saved to wikiart_index.faiss")
