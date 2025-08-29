import torch
from torch import nn

class LoRABlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)

        self.rank = rank
        self.alpha = alpha

    def forward(self, x):
        return (self.alpha / self.rank) * self.lora_b(self.lora_a(x))

class LoRALinear(torch.nn.Module):
    def __init__(self, linear, rank, alpha, lora_dropout=0.):
        super().__init__()
        self.linear = linear
        self.lora = LoRABlock(linear.in_features, linear.out_features, rank, alpha)
        self.dropout = nn.Dropout(p=lora_dropout)

    def forward(self, x):
        return self.linear(x) + self.dropout(self.lora(x))
