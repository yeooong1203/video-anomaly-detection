import torch
import torch.nn as nn

class ResidualAdapter2048(nn.Module):
    def __init__(self, d=2048, use_ln=True):
        super().__init__()
        self.delta = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.delta.weight)

        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        y = x + self.delta(x)   # residual
        if self.use_ln:
            y = self.ln(y)
        return y