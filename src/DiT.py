import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from src.noise_scheduler import get_schedule, add_noise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def positional_encoding(t, dim, period=10000):
    emb = torch.zeros(1, dim)
    for i in range(0, dim, 2):
        emb[0, i] = np.sin(t / period ** ((i) / dim))
        emb[0, i + 1] = np.cos(t / period ** ((i) / dim))
        
    return torch.tensor(emb)


class patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    def forward(self, x): #input shape: (B, C, W, H)
        batch_size, n_channels, self.width, self.height = x.size()
        self.C = n_channels
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # shape (B, C, W//P, H//P, P, P)
        patches = patches.contiguous().view(batch_size, n_channels, -1, self.patch_size, self.patch_size).permute(0, 2, 1, 3, 4)
        # shape (B, N, C, P, P)
        patches = patches.contiguous().view(batch_size, -1, n_channels * self.patch_size * self.patch_size)
        # shape (B, N, C*P*P)
        return patches

    def unpatchify(self, x): #input shape: (B, N, C*P*P)
        batch_size, num_patches, flattened_dim = x.size()
        patch_size = self.patch_size
        num_channels = self.C
        # Calculate grid dimensions
        grid_size = int(num_patches ** 0.5)
        # Reshape and permute
        x = x.view(batch_size, grid_size, grid_size, num_channels, patch_size, patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        # Combine patches
        x = x.view(batch_size, num_channels, grid_size * patch_size, grid_size * patch_size)
        return x
    
class DiT(nn.Module):
    def __init__(self, dims=512):
        super(DiT, self).__init__()
        self.patchify = patchify(4)
        self.embd = nn.Linear(512, dims)
        self.fc1 = nn.Linear(dims, dims)
        self.fc2 = nn.Linear(dims, dims)
        self.silu = nn.SiLU()
        self.dims = dims
        self.norm = nn.LayerNorm(dims)
        self.time_emb_layer = nn.Sequential(
            nn.Linear(dims, dims),
            nn.SiLU(),
            nn.Linear(dims, dims),
        )
        self.blocks = nn.ModuleList([DIT_block(dims) for _ in range(6)])
        self.linear = nn.Linear(dims, dims * 2)

    def forward(self, x, t): #input shape: (B, N, C, W, H)
        x = self.patchify(x)
        x = self.embd(x) 
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        # X shape: B, N, D
        spatial_embedding = [positional_encoding(i, self.dims)[0] for i in range(x.size(1))]
        spatial_embedding = torch.stack(spatial_embedding).unsqueeze(0).repeat(x.size(0), 1, 1).to(device)
        x = x + spatial_embedding

        noise_embedding = positional_encoding(t, self.dims).to(device).unsqueeze(0).repeat(x.size(0), 1, 1)
        noise_embedding = self.time_emb_layer(noise_embedding)
        x = torch.concat((x, noise_embedding), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.linear(x)
        x = x[:, :-1, :]
        x = x.contiguous()
        noise, var = torch.chunk(x, 2, dim=2)
        noise, var = self.patchify.unpatchify(noise), self.patchify.unpatchify(var)
        # Uncomment this line later 
        # noise = (noise + torch.randn(var.size()).to(device) * var)
        return noise
    
class DIT_block(nn.Module):
    def __init__(self, dims):
        super(DIT_block, self).__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.attn = nn.MultiheadAttention(dims, num_heads=4, batch_first=True)
        self.linear = nn.Linear(dims, dims)

    def forward(self, x):
        residue = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x += residue
        residue = x
        x = self.norm2(x)
        x = self.linear(x)
        x += residue
        return x

def train_denoiser(dit_model, vae, dataloader, num_steps=10, lr=0.0001, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dit_model.train()
    vae.eval()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dit_model.parameters(), lr=lr)
    alphas = get_schedule(num_steps)
    losses = []
    losses_mean = []
    for epoch in range(num_epochs + 1):
        for i, input in enumerate(dataloader):
            input = input[:, 0].to(device)
            with torch.no_grad():
                input = vae.encoder(input)
            t = np.random.randint(0, num_steps - 1)
            alpha = torch.tensor(alphas[t])
            noised = add_noise(input, alpha).to(device)

            optimizer.zero_grad()
            loss = criterion(dit_model(noised, t), noised - input)
            losses.append(loss.item())
            loss_mean = torch.tensor(losses[-40:]).mean()
            losses_mean.append(loss_mean)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss_mean:.4f}')
        if epoch % 5 == 0:
            clear_output()
            plt.plot(losses)
            plt.plot(losses_mean)
            plt.show()

def diffusion_denoising(vae, dit_model, num_steps=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.eval()
    dit_model.eval()
    img = dataset[0][0].unsqueeze(0).to(device)
    latent = vae.encoder(img)
    latent = torch.randn(latent.size()).to(device)
    output = vae.decoder(latent)
    alphas = torch.tensor(get_schedule(num_steps))
    plt.figure(figsize=(3, 3))
    plt.imshow(output[0].permute(1, 2, 0).cpu().detach().numpy())
    plt.axis('off')
    plt.show()
    time.sleep(0.1)
    with torch.no_grad():
        for t in range(num_steps-1, -1, -1):
            print(f'Step {t}, alpha: {alphas[t]}')
            eps =  dit_model(latent, t)
            sigma_t = torch.sqrt(1 - alphas[t])
            z = torch.randn_like(latent).to(device) if t > 0 else 0
            # denoise from z_t to z_0 but also add some noise to get z_(t-1)  [i think thats how it works] 
            latent = (latent - torch.sqrt(1 - alphas[t]) * eps) / torch.sqrt(alphas[t]) 
            latent = latent * torch.sqrt(alphas[t]) + sigma_t * z
            #latent = latent - eps
            output = vae.decoder(latent)
            output = torch.clamp(output, 0, 1)
            plt.figure(figsize=(3, 3))
            plt.imshow(output[0].permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            plt.show()