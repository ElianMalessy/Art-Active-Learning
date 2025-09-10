from torch.utils.data import Dataset
import torch
import glob

class EmbeddingDataset(Dataset):
    def __init__(self, save_dir):
        self.files = sorted(glob.glob(f"{save_dir}/batch_*.pt"))
        self.index_map = []

        # Precompute index mapping (which file + which row)
        for file_idx, f in enumerate(self.files):
            data = torch.load(f, map_location="cpu")
            n = data['embeddings'].shape[0]
            self.index_map.extend([(file_idx, i) for i in range(n)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, row_idx = self.index_map[idx]
        data = torch.load(self.files[file_idx], map_location="cpu")
        embedding = data['embeddings'][row_idx]
        prompt = data['prompts'][row_idx]
        return embedding, prompt
