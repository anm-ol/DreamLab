import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from IPython.display import clear_output
from src.blocks import residualBlock, downSample, upSample, attentionBlock

class vae(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            downSample(3, 16, use_batchnorm=False),
            residualBlock(16, 16),
            downSample(16, 16),
            residualBlock(16, 32, use_batchnorm=False),
            downSample(32, 32),
            residualBlock(32, 32),
            residualBlock(32, 32),
            downSample(32, 32, use_batchnorm=False),
            residualBlock(32, 32),
            attentionBlock(32), 
            residualBlock(32, 32),
            #nn.Linear(8 * 8 * 64, latent_dim)
        )
        self.decoder = nn.Sequential(
            #nn.Linear(latent_dim, 8 * 8 * 64),
            residualBlock(32, 32),
            attentionBlock(32),
            residualBlock(32, 32),
            upSample(32, 32),
            residualBlock(32, 32),
            residualBlock(32, 32),
            upSample(32, 32),
            residualBlock(32, 16, use_batchnorm=False),
            upSample(16, 16),
            residualBlock(16, 16),
            upSample(16, 3)
        )
        self.loss_func = nn.MSELoss()
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.drop = nn.Dropout(0.2)

    def encode(self, x):
        return self.drop(self.encoder(x))
    def decode(self, x):
        return self.decoder(x)
    def forward(self, x):
        latent = self.encoder(x)
        #latent = self.bn1(latent)
        #latent = self.drop(latent)
        out = self.decoder(latent)
        with torch.no_grad():
            pass
            #reconstructed_features = self.encoder(out)
        loss = self.loss_func(out, x) #+ perceptual_loss(latent, reconstructed_features, beta=1)
        return loss, out, latent
    
class VAE2(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE2, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * latent_dim) # For mean and logvar
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256*4*4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()  # To output values between 0 and 1
        )

    def encode(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        loss = self.loss_function(recon, x, mean, logvar)
        return loss, recon, z
    
    def loss_function(self, recon_x, x, mean, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.shape[0]
        return recon_loss + 0.000 * kl_loss

class vaeKL(nn.Module):
    def __init__(self, latent_dim=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            downSample(3, 16, use_batchnorm=False),
            residualBlock(16, 16),
            downSample(16, 16),
            residualBlock(16, 32, use_batchnorm=False),
            downSample(32, 32),
            residualBlock(32, 32),
            residualBlock(32, 32),
            downSample(32, 32, use_batchnorm=False),
            residualBlock(32, 32),
            attentionBlock(32), 
            residualBlock(32, 32),
            nn.Flatten(),
            nn.Linear(16* 16 * 32, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16 * 16 * 32),
            nn.Unflatten(1, (32, 16, 16)),
            residualBlock(32, 32),
            attentionBlock(32),
            residualBlock(32, 32),
            upSample(32, 32),
            residualBlock(32, 32),
            residualBlock(32, 32),
            upSample(32, 32),
            residualBlock(32, 16, use_batchnorm=False),
            upSample(16, 16),
            residualBlock(16, 16),
            upSample(16, 3)
        )
        self.loss_func = nn.MSELoss()
        self.bn1 = nn.BatchNorm1d(2*latent_dim)
        self.drop = nn.Dropout(0.2)

    def encode(self, x):
        return self.drop(self.encoder(x))
    def decode(self, x):
        return self.decoder(x)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        latent = self.encoder(x)
        latent = self.bn1(latent)
        mean, log_var = torch.chunk(latent, 2, dim=1)
        log_var = torch.clamp(log_var, min=-10, max=10)
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z)
        loss = self.loss_func(out, x) #+ perceptual_loss(latent, reconstructed_features, beta=1)
        kl_div_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
        loss += 0.001 * kl_div_loss
        return loss , out, latent

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4]).eval()
        for param in self.slice1.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg, y_vgg = self.slice1(x), self.slice1(y)
        loss = nn.functional.l1_loss(x_vgg, y_vgg)
        return loss


def train(model, data_loader, dataset=None, lr=None, optimizer=None, 
          device='cuda', num_epochs=100, perceptual_loss=True):
    losses = []
    losses_mean = []
    if perceptual_loss:
        perceptual = PerceptualLoss().to(device)
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr)
    for epoch in range(num_epochs + 1):
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            input = batch.to(device)
            if input.dim() == 5:
                _, _, c, h, w = input.shape
                input = input.view(-1, c, h, w)
            loss, reconstructed_image, _ = model(input)
            if perceptual_loss:
                loss += perceptual(input, reconstructed_image)
            loss.backward()
            losses.append(loss.item())
            loss_mean = torch.tensor(losses[-20:-1]).mean()
            losses_mean.append(loss_mean)
            optimizer.step()

            if i % 1000 == 0:
                pass
                print(f'loss: {loss_mean:.4f}, epoch: {epoch}, batch: {i}')
                pass

        if epoch % 5 == 0:
            clear_output()
            plt.plot(losses)
            plt.plot(losses_mean)
            if dataset:
                sample_image(model, dataset)
            plt.show()
    

def sample_image(model, dataset, device=None, generator=None):
    model.eval()
    device = device or 'cuda'
    with torch.no_grad():
        torch.manual_seed(generator) if generator else None
        
        random_idx = np.random.randint(0, len(dataset))
        random_idx = torch.randint(0, len(dataset), (1,)).item()
        random_image = dataset[random_idx]
        if random_image.dim() == 5:
            _, _, c, h, w = random_image.shape
            random_image = random_image.view(-1, c, h, w)
        random_image = random_image.to(device)[0].unsqueeze(0)
        loss, reconstructed_image, latent_space = model(random_image)
        
        # Clip the data to the valid range [0, 1]
        random_image = torch.clamp(random_image, 0, 1)
        reconstructed_image = torch.clamp(reconstructed_image, 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
        
        # Original image
        axes[0].imshow(random_image.squeeze(0).transpose(2, 0).transpose(0, 1).cpu())
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Reconstructed image
        axes[1].imshow(reconstructed_image.squeeze(0).transpose(2, 0).transpose(0, 1).cpu())
        axes[1].set_title('Reconstructed Image')
        axes[1].axis('off')
        
        plt.show()