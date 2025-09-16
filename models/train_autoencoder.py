import torch
import torch.nn.functional as F

from utils import device
from models.autoencoder import Encoder, Decoder, rbf_mmd_efficient, mmd_unbiased_multi_sigma

def pairwise_cosine_matrix(x, eps=1e-8):
    x_norm = F.normalize(x, dim=1, eps=eps)
    return x_norm @ x_norm.t()

def cosine_preservation_loss(x, x_hat):
    Cx = pairwise_cosine_matrix(x)
    Cx_hat = pairwise_cosine_matrix(x_hat)
    return F.mse_loss(Cx, Cx_hat)

def train_autoencoder(dataloader, num_epochs=100, patience=5):
    encoder_model = Encoder().to(device)
    decoder_model = Decoder().to(device)

    encoder = torch.compile(encoder_model)
    decoder = torch.compile(decoder_model)

    optimizer = torch.optim.Adam(params=list(encoder_model.parameters())+list(decoder_model.parameters()), lr=1e-4)

    num_batches = len(dataloader)
    epochs_no_improve = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        epoch_preservation = 0.0
        epoch_mmd = 0.0
        for clip_embeddings in dataloader:
            x = clip_embeddings.to(device)
            optimizer.zero_grad()

            z = encoder(x)
            x_hat = decoder(z)

            z_prior = torch.randn_like(z).to(device)
            lambda_mmd = 0.5
            cosine_loss = cosine_preservation_loss(x, x_hat) 
            mmd_loss = mmd_unbiased_multi_sigma(z, z_prior, torch.tensor([0.5,1.0,2.0,4.0]))*lambda_mmd
            loss = cosine_loss + mmd_loss

            loss.backward()
            optimizer.step()

            epoch_preservation += cosine_loss.item()
            epoch_mmd += mmd_loss.item()

        avg_preservation = epoch_preservation / num_batches
        avg_mmd = epoch_mmd / num_batches
        avg_loss = avg_preservation + avg_mmd

        print(f"Average mmd loss: {avg_mmd}")
        print(f"Average preservation: {avg_preservation}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(encoder_model.state_dict(), "encoder_model.pth")
            print("Model improved and saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break


