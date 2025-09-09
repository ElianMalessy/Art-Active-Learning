from dataset import WikiArtDataset
from torchvision import transforms
from setup import store_embeddings
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--clear', action='store_true')

if __name__ == '__main__':
    if parser.parse_args().clear:
        dataset = load_dataset("huggan/wikiart", split="train")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # CLIP default
            # no ToTensor needed if you pass PIL images to CLIPProcessor
        ])

        dataset = WikiArtDataset(dataset, transform)
        store_embeddings(dataset)



