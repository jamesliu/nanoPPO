# filename: sinusoidal_positional_encoding.py

import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device=torch.device("cpu")):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        div_term = torch.repeat_interleave(div_term, 2, dim=0)  # Repeat each element twice
        self.encoding[:, 0::2] = torch.sin(position * div_term[:d_model//2])
        self.encoding[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()
