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
            attentionBlock(16), 

            downSample(16, 16),
            residualBlock(16, 32, use_batchnorm=False),
            attentionBlock(32), 

            downSample(32, 32),
            residualBlock(32, 32),
            attentionBlock(32), 

            downSample(32, 32, use_batchnorm=False),
            residualBlock(32, 32),
            attentionBlock(32), 
            residualBlock(32, 32),
            attentionBlock(32), 
            residualBlock(32, 32),

        )
        self.decoder = nn.Sequential(
            residualBlock(32, 32),
            attentionBlock(32),
            residualBlock(32, 32),
            attentionBlock(32),
            residualBlock(32, 32),
            upSample(32, 32),

            attentionBlock(32),
            residualBlock(32, 32),
            upSample(32, 32),

            attentionBlock(32),
            residualBlock(32, 16, use_batchnorm=False),
            upSample(16, 16),

            attentionBlock(32),
            residualBlock(16, 16),
            upSample(16, 3)
        )
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.loss_func = nn.L1Loss()

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode(self, x):
        return self.encoder(x)
    def decode(self, x):
        return self.decoder(x)
    def forward(self, x):
        latent  = self.encoder(x)
        b, c, h, w = latent.shape
        #mean, logvar = torch.chunk(latent, 2, dim=1)
        #latent = self.reparameterize(mean, logvar)
        latent = latent.view(b, c, -1)
        latent = self.fc1(latent)
        latent = nn.SiLU()(latent)
        latent = self.fc2(latent)
        mean, logvar = torch.chunk(latent, 2, dim=2)
        latent = self.reparameterize(mean, logvar).view(b, c, h, w)

        out = self.decoder(latent)

        loss = self.loss_func(out, x)
        #kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.shape[0]
        #loss += 5e-6 * kl_loss
        return loss, out, latent

class Disciminator(nn.Module):
    def __init__(self, im_channels=3,
                 conv_channels=[64, 128, 256],
                 kernels=[4,4,4,4],
                 strides=[2,2,2,1],
                 paddings=[1,1,1,1]):
        super().__init__()
        self.im_channels = im_channels
        activation = nn.LeakyReLU(0.2)
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False if i !=0 else True),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

class VAE2(nn.Module):
    def __init__(self, latent_dim=32, in_channels=3, hidden_dims=[16, 32, 32]):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = 0.001  # Beta-VAE weight
        
        # Encoder layers
        encoder_layers = []
        current_channels = in_channels
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                downSample(current_channels, h_dim),
                residualBlock(h_dim, h_dim),
                residualBlock(h_dim, h_dim),
                attentionBlock(h_dim)
            ])
            current_channels = h_dim
            
        self.encoder_backbone = nn.Sequential(*encoder_layers)
        
        # Latent space projections
        self.mean_proj = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], latent_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 1)
        )
        
        self.logvar_proj = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], latent_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 1)
        )
        
        # Decoder layers
        decoder_layers = []
        hidden_dims.reverse()
        current_channels = latent_dim
        
        for h_dim in hidden_dims:
            decoder_layers.extend([
                upSample(current_channels, h_dim),
                residualBlock(h_dim, h_dim),
                residualBlock(h_dim, h_dim),
                attentionBlock(h_dim)
            ])
            current_channels = h_dim
            
        # Final output layer
        decoder_layers.extend([
            nn.Conv2d(hidden_dims[-1], in_channels, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean
    
    def encode(self, x):
        features = self.encoder_backbone(x)
        mean = self.mean_proj(features)
        logvar = self.logvar_proj(features)
        return mean, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):

        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        
        recon_loss = F.l1_loss(recon, x, reduction='sum') / x.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.shape[0]
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return total_loss, recon, z

    def sample(self, num_samples, device='cuda'):
        z = torch.randn(num_samples, self.latent_dim, 8, 8).to(device)  # Adjust spatial dims as needed
        samples = self.decode(z)
        return samples
    
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

class vae3d(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        return x

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
          device='cuda', num_epochs=100, perceptual_loss=True, disc_start=None):
    losses = []
    losses_mean = []
    if perceptual_loss:
        perceptual = PerceptualLoss().to(device)
    if disc_start is not None:
        discriminator = Disciminator().to(device)
        discriminator.train()
        disc_criteria = nn.BCEWithLogitsLoss()
        optim_disc = torch.optim.Adam(discriminator.parameters(), lr=1e-5)
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr)

    step_count = 0
    for epoch in range(num_epochs + 1):
        for i, batch in enumerate(data_loader):
            model.train()

            optimizer.zero_grad()
            if disc_start:
                optim_disc.zero_grad()

            input = batch.to(device)
            if input.dim() == 5:
                _, _, c, h, w = input.shape
                input = input.view(-1, c, h, w)
            loss, reconstructed_image, _ = model(input)
            if perceptual_loss:
                loss += 1 * perceptual(input, reconstructed_image)

            # generator loss
            if disc_start is not None and step_count > disc_start:
                fake_pred = discriminator(reconstructed_image)
                fake_loss = disc_criteria(fake_pred, torch.ones_like(fake_pred).to(device))
                loss += 0.1 * fake_loss
            loss.backward()
            losses.append(loss.item())
            loss_mean = torch.tensor(losses[-20:-1]).mean()
            losses_mean.append(loss_mean)
            
            # discriminator loss
            if disc_start is not None and step_count > disc_start:
                real_pred = discriminator(input)
                real_loss = disc_criteria(real_pred, torch.ones_like(real_pred).to(device))
                fake_pred = discriminator(reconstructed_image.detach())
                fake_loss = disc_criteria(fake_pred, torch.zeros_like(fake_pred).to(device))
                disc_loss = 0.5 * (real_loss + fake_loss)
                disc_loss.backward()
                optim_disc.step()

            optimizer.step()
        
            step_count += 1
        
        optimizer.step()
        optimizer.zero_grad()
        if disc_start:
            optim_disc.step()
            optim_disc.zero_grad()
        
        print(f'loss: {loss_mean:.4f}, epoch: {epoch}, batch: {i}')

        if epoch % 5 == 0:
            clear_output()
            plt.plot(losses)
            plt.plot(losses_mean)
            if dataset:
                sample_image(model, dataset)
                sample_image(model, dataset)
                sample_image(model, dataset)
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
        min, max = random_image.min(), random_image.max()
        range = max - min
        random_image = (random_image - min) / range
        reconstructed_image = ((reconstructed_image + 1)/2).clamp(0, 1)

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