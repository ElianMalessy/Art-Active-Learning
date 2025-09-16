import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import argparse

from dataset.wikiart import WikiArtDataset
from dataset.embeddings import EmbeddingDataset
from storage import store_clip_embeddings, store_latent_embeddings
from models.train_autoencoder import train_autoencoder
from models.autoencoder import Encoder
from utils import device

def generate_clip_embeddings():
    """Generate CLIP embeddings with metadata (run once)"""
    print("Loading WikiArt dataset...")
    wikiart_dataset = load_dataset('huggan/wikiart', split='train')

    print("Creating dataset with transforms...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = WikiArtDataset(wikiart_dataset, transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    print("Storing CLIP embeddings with metadata...")
    store_clip_embeddings(dataloader, 'wikiart_embeddings')
    print("CLIP embeddings generated!")

def train_autoencoder_model():
    """Train the autoencoder (run when adjusting autoencoder)"""
    print("Loading CLIP embeddings for autoencoder training...")
    dataset = EmbeddingDataset('wikiart_embeddings')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    print("Training autoencoder...")
    train_autoencoder(dataloader)
    print("Autoencoder training completed!")

def generate_latent_embeddings():
    """Generate latent embeddings with metadata (run after autoencoder training)"""
    print("Loading CLIP embeddings for latent generation...")
    dataset = EmbeddingDataset('wikiart_embeddings')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    print("Generating latent embeddings with metadata...")
    encoder = Encoder()
    encoder.load_state_dict(torch.load('encoder_model.pth'))
    encoder = encoder.to(device)
    encoder.eval()
    store_latent_embeddings(dataloader, encoder, 'latent_embeddings', 
                           source_metadata_path='wikiart_embeddings/image_metadata.pkl')
    print("Latent embeddings generated!")

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings with flexible workflow')
    parser.add_argument('--clip', action='store_true', 
                       help='Generate CLIP embeddings (run once)')
    parser.add_argument('--train', action='store_true', 
                       help='Train autoencoder (run when adjusting model)')
    parser.add_argument('--latent', action='store_true', 
                       help='Generate latent embeddings (run after training)')
    parser.add_argument('--all', action='store_true', 
                       help='Run entire pipeline (clip + train + latent)')
    
    args = parser.parse_args()
    
    if args.all:
        print("Running entire pipeline...")
        generate_clip_embeddings()
        train_autoencoder_model()
        generate_latent_embeddings()
        print("Complete pipeline finished!")
    elif args.clip:
        generate_clip_embeddings()
    elif args.train:
        train_autoencoder_model()
    elif args.latent:
        generate_latent_embeddings()
    else:
        print("No action specified. Use --help for options.")
        print("\nTypical workflow:")
        print("1. python regenerate_embeddings.py --clip    # Generate CLIP embeddings (once)")
        print("2. python regenerate_embeddings.py --train   # Train autoencoder (iterative)")
        print("3. python regenerate_embeddings.py --latent  # Generate latents (after training)")

if __name__ == '__main__':
    main()
