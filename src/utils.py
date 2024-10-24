import torch
import cv2
from src.DiT import *

def count_parameters(model):
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad) * 1e-6}M')
    
def save_model_with_config(dit, path, config, vae=None):
    # Save the model's state_dict and config together
    checkpoint = {
        'model_state_dict': dit.state_dict(),  # Save model weights
        'config': config  # Save the configuration
    }

    # Save to a file
    torch.save(checkpoint, path)

def play_video(video_path, start_frame=0, end_frame=None):
    # will play the video embedded in the notebook
    pass

def load_model_with_config(path):
    # Load the model's state_dict and config together
    checkpoint = torch.load(path)
    config = checkpoint['config']
    # Extract values from the config
    num_dit_blocks = config.get('num_dit_blocks')
    patch_size = config.get('patch_size')
    dims = config.get('dims')

    # Initialize the model with the extracted config values
    model = DiT(num_dit_blocks, patch_size, dims)
    # Load model weights
    dit_model = DiT()
    model.load_state_dict(checkpoint['model_state_dict'])
    # Load the configuration
    config = checkpoint['config']
    print(f"Model loaded from {path}")
    print(f"Configuration: {config}")
    return model, config