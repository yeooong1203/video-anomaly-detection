import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
import option
import math 
from tqdm import tqdm

args = option.parser.parse_args()


def adaptive_hybrid_loss(outputs, targets, threshold=0.3):
    error = torch.abs(outputs - targets)
    
    bce = F.binary_cross_entropy(outputs, targets, reduction='none')
    hb = F.huber_loss(outputs, targets, delta=0.3, reduction='none')
    
    # Error 작으면 huber, 크면 BCE
    mask = (error < threshold).float()
    
    loss = mask * hb + (1 - mask) * bce
    return loss



def concatenated_train_variable_length(train_loader, model, optimizer, epoch, 
                                      device, use_packed_lstm=False,
                                      accumulation_steps=1):

    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    
    for batch_idx, (features, labels, masks, lengths) in enumerate(pbar):
        features = features.to(device)  # (B, max_T, 2048)
        labels = labels.to(device)      # (B, max_T) - padding=-1
        masks = masks.to(device)        # (B, max_T)
        
        B, T, D = features.shape
        
        outputs = model(features)
        outputs = outputs.squeeze(-1)  # (B, T)

        
        loss = adaptive_hybrid_loss(
            outputs, labels, threshold=0.3
        )  # (B, T)
        
        loss = loss * masks

        num_valid = masks.sum()
        if num_valid > 0:
            loss = loss.sum() / num_valid
        else:
            loss = torch.tensor(0.0, device=device)
        
        loss = loss / accumulation_steps
        
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
    if num_batches % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches
    
    return avg_loss