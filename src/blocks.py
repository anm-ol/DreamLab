from torch import nn 
import numpy as np
class residualBlock(nn.Module):
    def __init__(self, ni, nf, use_batchnorm=True): 
        super().__init__()
        self.ni = ni
        self.nf = nf
        self.activation = nn.GELU()
        self.conv1 = nn.Conv2d(ni, nf, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nf) if use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(nf) if use_batchnorm else nn.Identity()
        if ni == nf:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(ni, nf, kernel_size=1, stride=1)
    def forward(self, x):
        residue = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + self.skip(residue)
        x = self.activation(x)
        return x

class downSample(nn.Module):
    def __init__(self, ni, nf, use_batchnorm=True):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
        self.activation = nn.GELU()
        self.bn = nn.BatchNorm2d(nf) if use_batchnorm else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
class upSample(nn.Module):
    def __init__(self, ni, nf, use_batchnorm=True):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=3, stride=1, padding=1)
        self.activation = nn.GELU()
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn = nn.BatchNorm2d(nf) if use_batchnorm else nn.Identity()
    def forward(self, x):
        x = self.upSample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    def forward(x, patch_size):
        batch_size, n_channels, width, height = x.size()
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, n_channels, -1, patch_size, patch_size).permute(0, 2, 1, 3, 4)
        #patches = patches.contiguous().view(batch_size, -1, n_channels * patch_size * patch_size)
        return patches
    
class attentionBlock(nn.Module):
    def __init__(self, n_emb, n_heads=4):
        super().__init__()
        self.flatten = nn.Flatten(2)
        #self.n_input = n_input
        self.n_emb = n_emb
        self.norm = nn.GroupNorm(4, n_emb)
        self.attention = nn.MultiheadAttention(n_emb, n_heads, bias=True,  batch_first=True)

    def forward(self, x):
        batch_size, n_channels, h, w = x.size()
        residue = x
        x = self.norm(x)
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1).view(batch_size, n_channels, h, w)
        
        return x + residue
  