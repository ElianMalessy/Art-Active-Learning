from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import argparse

from dataset.wikiart import WikiArtDataset
from setup import store_embeddings
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--clear', action='store_true')

if __name__ == '__main__':
    wikiart_dataset = load_dataset("huggan/wikiart", split="train")

    if parser.parse_args().clear:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # CLIP default
            # no ToTensor needed if you pass PIL images to CLIPProcessor
        ])

        dataset = WikiArtDataset(wikiart_dataset, transform)
        batch_size = 128
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        store_embeddings(dataset)

    train()



