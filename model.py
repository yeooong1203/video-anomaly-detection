import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F
import math
#torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Model_V2_AllCNN(nn.Module):
    
    def __init__(self, n_features, kernel_size=5):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_features, 256, kernel_size, padding=kernel_size//2)
        
        self.conv_att1 = nn.Conv1d(n_features, 256, kernel_size, padding=kernel_size//2)
        
        self.conv2 = nn.Conv1d(256, 64, kernel_size, padding=kernel_size//2)
        
        self.conv_att2 = nn.Conv1d(256, 64, kernel_size, padding=kernel_size//2)
        
        self.fc_out = nn.Linear(64, 1)
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        orig_shape = inputs.shape
        
        if inputs.dim() == 3:
            B, T, D = inputs.shape
            use_temporal = True
        else:
            B = inputs.shape[0]
            T = 1
            inputs = inputs.unsqueeze(1)
            use_temporal = False
        
        # (B, T, D) → (B, D, T)
        x = inputs.permute(0, 2, 1)  # (B, 2048, T)
        
        att1 = torch.sigmoid(self.conv_att1(x))  # (B, 512, T)
        x = self.conv1(x)                         # (B, 512, T)
        x = x * att1 + att1  # Gated attention
        x = self.gelu(x)
        x = self.dropout1(x)
        
        att2 = torch.sigmoid(self.conv_att2(x))  # (B, 32, T)
        x = self.conv2(x)                         # (B, 32, T)
        x = x * att2 + att2
        x = self.gelu(x)
        x = self.dropout2(x)
        
        # (B, T, 32)
        x = x.permute(0, 2, 1)
        
        x = self.sigmoid(self.fc_out(x))

        x = x.permute(0, 2, 1)   # (B, 1, T)

        x = F.avg_pool1d(
            x,
            kernel_size=7,
            stride=1,
            padding=3
        )

        x = x.permute(0, 2, 1)   # (B, T, 1)
        
        return x
