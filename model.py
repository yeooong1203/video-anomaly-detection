import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_V2_AllCNN(nn.Module):
    
    def __init__(self, n_features, kernel_size=5):
        super().__init__()
        '''
        self.conv1 = nn.Conv1d(n_features, 128, kernel_size=3, padding=3//2)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv_att1 = nn.Conv1d(n_features, 128, kernel_size=3, padding=3//2)
        
        self.conv2 = nn.Conv1d(128, 32, kernel_size=5, padding=5//2)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv_att2 = nn.Conv1d(128, 32, kernel_size=5, padding=5//2)
        
        self.fc_out = nn.Linear(32, 1)
        '''
        self.conv1 = nn.Conv1d(n_features, 256, kernel_size=3, padding=3//2)
        #self.conv1 = nn.Conv1d(n_features, 256, kernel_size=3, padding=3//2, padding_mode='replicate')
        self.bn1 = nn.BatchNorm1d(256)

        self.conv_att1 = nn.Conv1d(n_features, 256, kernel_size=3, padding=3//2)
        
        self.conv2 = nn.Conv1d(256, 64, kernel_size=5, padding=5//2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv_att2 = nn.Conv1d(256, 64, kernel_size=5, padding=5//2)
        
        self.fc_out = nn.Linear(64, 1)
        
        self.dropout1 = nn.Dropout1d(0.2)
        self.dropout2 = nn.Dropout1d(0.2)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        

        # auc 82 나왔던 모델 돌리려면 필요! alpha ..
        #self.alpha = nn.Parameter(torch.ones(1))
    
    def forward(self, inputs, return_logits=False):
        if inputs.dim() != 3:
            inputs = inputs.unsqueeze(1)
            use_temporal = False
        
        # (B, T, D) → (B, D, T)
        x = inputs.permute(0, 2, 1)  # (B, 2048, T)
        
        att1 = torch.sigmoid(self.conv_att1(x))  # (B, 256, T)
        x = self.conv1(x)                         # (B, 256, T)
        x = x * att1 + att1  # Gated attention
        x = self.gelu(x)
        #x = x + att1
        x = self.dropout1(x)
        '''x = x.unsqueeze(-1) # (B, C, T, 1)
        x = self.spatial_dropout(x)
        x = x.squeeze(-1) # (B, C, T)'''
        
        att2 = torch.sigmoid(self.conv_att2(x))  # (B, 64, T)
        x = self.conv2(x)                         # (B, 64, T)
        x = x * att2 + att2
        x = self.gelu(x)
        #x = x + att2
        x = self.dropout2(x)
        '''x = x.unsqueeze(-1) # (B, C, T, 1)
        x = self.spatial_dropout(x)
        x = x.squeeze(-1) # (B, C, T)'''
        
        # (B, T, 64)
        x = x.permute(0, 2, 1)
        
        logits = self.fc_out(x) 
        probs = self.sigmoid(logits)
       
        probs = probs.permute(0, 2, 1) # (B, 1, T)
        probs = F.avg_pool1d(
            probs,
            kernel_size=7,
            stride=1,
            padding=3
        )
        probs = probs.permute(0, 2, 1)   # (B, T, 1)
       
        if return_logits:
            
            logits_pooled = logits.permute(0, 2, 1)
            logits_pooled = F.avg_pool1d(
                logits_pooled,
                kernel_size=7,
                stride=1,
                padding=3
            )
            logits_pooled = logits_pooled.permute(0, 2, 1)
            
            return probs, logits_pooled
        
        return probs
