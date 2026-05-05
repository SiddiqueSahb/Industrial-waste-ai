"""
Load model from disk and run predictions on a image
"""

#Load the Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from src.classification.models import get_model
from PIL import Image

#Load device
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

#load Trained model
def load_trained_model(model_name: str, weight_path: str | Path , num_classes: int = 28,dropout: float = 0.2,
device: torch.device = device()) -> nn.Module:
    #Set device
    device = device or device()
    #load model
    model = get_model(model_name, num_classes=num_classes,pretrained=False, dropout=dropout)
    #load weights with full checkpoints not only weights
    state = torch.load(str(weight_path), map_location= "cpu",weights_only=False)
    #It loads model weights safely, even if they were saved in different formats.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    #load state dict
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_trained_model] missing keys ({len(missing)}): {missing[:3]}…")
    if unexpected:
        print(f"[load_trained_model] unexpected keys ({len(unexpected)}): {unexpected[:3]}…")
    #set model to evaluation mode   
    model.to(device).eval()
    #return model
    return model

#It takes an image , runs it through the model , returns class probabilities
@torch.no_grad()
def predict_image(model:nn.model,img:Image.Image,transform) -> np.ndarray:
    """Return class probabilities for the image"""
    #Whether model is in on cpu or gpu
    device = next(model.parameters()).device
    #Transform image
    img = transform(img)
    #Add batch dimension
    img = img.unsqueeze(0).to(device)
    #Run image through model and we get raw scores not probabilities
    output = model(img)
    #Return class probabilities only and not batch dimension
    #Convert raw scores → probabilities (sum = 1)
    #(batch_size, num_classes)
    #dim=1 → apply softmax across classes
    prob = F.softmax(output,dim=1).cpu().numpy()[0]
    return prob

#It takes an image , runs it through the model , returns topK class names and probabilities
@torch.no_grad()
def predict_image_topK(model:torch.nn.Module,img:Image.Image,transform,
                     class_names:List[str],topK:int = 5) -> List[Tuple[str,float]]:
        prob = predict_image(model,img,transform)
        #Get topK indices
        topK_indices = prob.argsort(prob)[::-1][:topK]
        return [(class_names[i],prob[i]) for i in topK_indices]