from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import cv2
import numpy as np

def empty_folder(folder):
    if not os.path.exists(folder):
        return
    files = os.listdir(folder)
    for file in files:
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def extract_frames(mp4_path, start_frame, end_frame, stride, save_dir='data/mario', 
                   test_split=0.2, size = (64,64)):
    video = cv2.VideoCapture(mp4_path)
    empty_folder(save_dir)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames in video:", total_frames)
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError("Invalid start frame number")
    if end_frame < 0 or end_frame >= total_frames:
        raise ValueError("Invalid end frame number")
    current_frame = start_frame
    frames = []
    while current_frame <= end_frame:
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = video.read()
        if not ret:
            raise ValueError("Error reading frame")
        frame = cv2.resize(frame, size)
        if np.random.rand() > test_split:
            cv2.imwrite(os.path.join(save_dir, "train", f"frame_{current_frame}.jpg"), frame)
        else:
            cv2.imwrite(os.path.join(save_dir, "test", f"frame_{current_frame}.jpg"), frame)
        cv2.imwrite(os.path.join(save_dir, f"frame_{current_frame}.jpg"), frame)
        frames.append(frame)
        current_frame += stride
    video.release()
    return frames

def split_data(data_dir, test_split=0.2, num_frames=4):
    files = os.listdir(data_dir)
    dir_idx = []
    for i, file in enumerate(files):
        if os.path.isdir(os.path.join(data_dir, file)):
            dir_idx.append(i)
            print("Removed directory:", file)

    for i in reversed(dir_idx):
        files.pop(i)

    os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)            

    file_groups = np.array_split(files, len(files)//num_frames)[1:]
    np.random.shuffle(file_groups)
    test_size = int(len(file_groups)*test_split)
    test_files = file_groups[:test_size]
    train_files = file_groups[test_size:]
    for files in train_files:
        for file in files:
            os.rename(os.path.join(data_dir, file), os.path.join(data_dir, "train", file))
    for files in test_files:
        for file in files:
            os.rename(os.path.join(data_dir, file), os.path.join(data_dir, "test", file))

def split_data2(data_dir, test_split=0.2, num_frames=4):
    files = os.listdir(data_dir)
    dir_idx = []
    for i, file in enumerate(files):
        if os.path.isdir(os.path.join(data_dir, file)):
            dir_idx.append(i)
            print("Removed directory:", file)

    for i in reversed(dir_idx):
        files.pop(i)

    file_groups = np.array_split(files, len(files)//num_frames)
    for i in range(len(file_groups)):
        if file_groups[i].shape[0] != num_frames:
            file_groups[i] = file_groups[i][:num_frames]

    np.random.shuffle(file_groups)
    test_size = int(len(file_groups)*test_split)
    test_files = file_groups[:test_size]
    train_files = file_groups[test_size:]
    
    with open ('data/train.txt', 'w') as f:
        for i, files in enumerate(train_files):
            f.write (f'{i} ')
            for file in files:
                f.write(f'{file} ')
            f.write('\n')
    
    with open ('data/test.txt', 'w') as f:
        for i, files in enumerate(test_files):
            f.write (f'{i} ')
            for file in files:
                f.write(f'{file} ')
            f.write('\n')

transform = transforms.Compose([
    transforms.Resize((64, 64)),    # Resize images to 128x128
    transforms.ToTensor(),            # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])


class videoDataset(Dataset):
    def __init__(self, data_dir, txt_file, resolution):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize(resolution),    # Resize images to 128x128
            transforms.ToTensor(),            # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        self.data = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.data.append((int(line[0]), line[1:]))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, files = self.data[idx]
        frames = []
        for file in files:
            img = Image.open(os.path.join(self.data_dir, file))
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        return torch.stack(frames)

class marioDataset(Dataset):
    def __init__(self, path_dir, num_frames=4, train=True, transform=None):
        self.dir = os.path.join(path_dir, "train" if train else "test")
        self.num_frames = num_frames
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.dir) if os.path.isfile(os.path.join(path_dir, f))]

    def __len__(self):
        return len(self.image_files)//self.num_frames 
    
    def __getitem__(self, idx):
        # Load the image from the file
        images = []
        for i in range(self.num_frames):
            img_path = os.path.join(self.dir, self.image_files[idx + i])
            with Image.open(img_path).convert("RGB") as image:
                image = self.transform(image)
                images.append(image)
        images = torch.stack(images)
        return images
    