import torch
import numpy as np

def get_schedule(n_steps, start=0.0, end=1):
    timesteps = np.linspace(start, end, n_steps)
    schedule = np.cos((timesteps*np.pi / (2/end) + 0.008) / 1.008)**2
    schedule = schedule/schedule[0]
    return schedule 

def add_noise(input, alpha): # input shape : (C, H, W)
    return input*torch.sqrt(alpha) + torch.randn_like(input)*torch.sqrt(1-alpha)

def get_noised_data(input): # input shape : (B, C, H, W)
    n_steps = 100
    alpha = get_schedule(n_steps)
    alpha = torch.from_numpy(get_schedule(n_steps)).float().to(input.device)
    alpha = alpha.view(-1, *([1]*len(input.shape)))
    input = input.view(1, *input.shape)
    input = input*np.sqrt(alpha) + torch.randn_like(input)*np.sqrt(1-alpha) # (n_steps, B, C, H, W)
    input = torch.clamp(input, 0., 1.) 
    return input