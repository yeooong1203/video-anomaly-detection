import torch
import torch.nn as nn

class DetectorInputLoRA(nn.Module):
    def __init__(self, d=2048, r=8, alpha=1.0):
        super().__init__()

        self.r = r
        self.alpha = alpha

        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)

        # LoRA initialization
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):

        delta = self.B(self.A(x))

        return x + self.alpha * delta