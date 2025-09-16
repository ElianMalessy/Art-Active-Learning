from torch.utils.data import Dataset

class WikiArtDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')

        if self.transform:
            image = self.transform(image)

        prompt = get_prompt(item, self.dataset)
        return image, prompt

def get_prompt(item, dataset):
    def label_to_names(field):
        value = item[field]
        feature = dataset.features[field]
        if isinstance(value, list):
            return ", ".join([feature.int2str(v) for v in value])
        else:
            return feature.int2str(value)

    genre = label_to_names('genre')
    artist = label_to_names('artist')
    style = label_to_names('style')

    prompt = f"A {genre} painting by {artist} in this style: {style}"
    return prompt
