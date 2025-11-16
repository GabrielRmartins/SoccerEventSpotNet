import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import numpy as np
from tqdm import tqdm
import json
import os
import h5py

class H5VideoDataset(Dataset): 
    def __init__(self, h5_path, labels_dict, transform=None, num_frames=13): 
        self.h5_path = h5_path 
        self.labels_dict = {} 
        self.keys = [] 
        self.transform = transform 
        self.num_frames = num_frames 
        # Open just valid processed video tensors from HDF5
        with h5py.File(self.h5_path, "r") as f: 
            for key, label in labels_dict.items(): 
                if key in f["data"] and f["data"][key].shape == (num_frames, 252, 252, 3): 
                    self.keys.append(key) 
                    self.labels_dict[key] = label 
                    
        
        print(f"{len(self.keys)} vídeos válidos encontrados de {len(labels_dict)}") 
    
    def __len__(self): return len(self.keys) 
        
    def __getitem__(self, idx): 
        key = self.keys[idx] 
        with h5py.File(self.h5_path, "r") as f: 
            data = np.array(f["data"][key], copy=True, dtype=np.float32) 

        # Data transformations as pre-trained models expected
        video = np.transpose(data, (3,0,1,2)) / 255.0 
        mean = np.array([0.45,0.45,0.45], dtype=np.float32) 
        std = np.array([0.225,0.225,0.225], dtype=np.float32) 
        video = (video - mean.reshape(3,1,1,1)) / std.reshape(3,1,1,1)
        
        if self.transform: 
            video = self.transform(video) 
            
        label = self.labels_dict[key] 
        
        return torch.from_numpy(video).float(), torch.tensor(label, dtype=torch.long)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for videos, labels in tqdm(loader, desc="Treinando", leave=False):
        videos, labels = videos.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * videos.size(0)
    return running_loss / len(loader.dataset)

def eval(model, loader, device, num_classes):
    model.eval()
    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)

    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Validando", leave=False):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            for i in range(num_classes):
                correct_per_class[i] += ((predicted == i) & (labels == i)).sum().item()
                total_per_class[i] += (labels == i).sum().item()

    acc_per_class = np.divide(correct_per_class, total_per_class, out=np.zeros_like(correct_per_class), where=total_per_class != 0)
    precision = np.divide(np.sum(correct_per_class),np.sum(total_per_class))
    mean_acc = np.mean(acc_per_class)
    return mean_acc, acc_per_class,precision

