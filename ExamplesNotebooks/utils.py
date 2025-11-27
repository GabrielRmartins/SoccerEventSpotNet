import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import numpy as np
from tqdm import tqdm
import json
import os
import h5py
import cv2

def prepare_input(video_path,num_frames=13, size=(252,252)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Unable to open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = 0
    end_frame = int(total_frames - 1)

    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        frames.append(frame)

    cap.release()

    data = np.array(frames, dtype=np.float32)

    video = np.transpose(data, (3,0,1,2)) / 255.0 
    mean = np.array([0.45,0.45,0.45], dtype=np.float32) 
    std = np.array([0.225,0.225,0.225], dtype=np.float32) 
    video = (video - mean.reshape(3,1,1,1)) / std.reshape(3,1,1,1)
    
    return torch.from_numpy(video).float()

def inference_single_video(model_path, video_tensor, device,labels_path,return_outputs=False):
    
    with open(os.path.join(labels_path,"idx_to_event.json"), "r") as f:
        idx_to_event = json.load(f)
    
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
    in_features = model.blocks[5].proj.in_features
    checkpoint = torch.load(model_path,weights_only = False, map_location=device)
    num_classes = checkpoint["model_state_dict"]["blocks.5.proj.weight"].shape[0]
    model.blocks[5].proj = nn.Linear(in_features, num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    video_tensor = video_tensor.unsqueeze(0).to(device)

    model.eval()
     
    with torch.no_grad():
        outputs = model(video_tensor)

    outputs = outputs.cpu().numpy()
    
    top_3 = np.argsort(outputs, axis=1)[0][-3:]

    top_3 = top_3[::-1]
    top_3_values = outputs[0][top_3]
    print(f"Top-{len(top_3)} Predictions:")
    for i in range(len(top_3)):
        print(f"Top {i} - Class: {idx_to_event[str(top_3[i])]}, Score: {top_3_values[i]:.4f}")
        
    if return_outputs:
        return outputs



def predict_waterfall_model(models, loader, device):
    labels_true = []
    labels_pred = []
    labels_root = []
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Evaluating Waterfall Model", leave=False):
            videos, labels = videos.to(device), labels.to(device)
            outputs = models["root"](videos)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                labels_root.append(predicted[i].item())
                labels_true.append(labels[i].item())
                current_model = models.get(predicted[i].item(), None)
                
                if current_model is not None:
                    
                    video_i = videos[i].unsqueeze(0)
                    output_i = current_model(video_i)
                    
                    labels_pred.append(output_i.cpu().numpy())
                else:
                    labels_pred.append(output_i.cpu().numpy())

    return labels_true, labels_pred, labels_root




def predict(model, loader, device):
    model.eval()
    expected_labels = []
    predicted_labels = []

    with torch.no_grad():
        
        for videos, labels in tqdm(loader, desc="Evaluating", leave=False):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)          
            expected_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(outputs.cpu().numpy())

        expected_labels = np.array(expected_labels)
        predicted_labels = np.array(predicted_labels)

    return expected_labels,predicted_labels