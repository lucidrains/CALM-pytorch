import torch
from torch import nn, einsum
from torch.nn import Module

from einops import rearrange, repeat

from x_transformers import (
    Attention
)

# helpers

def exists(v):
  return v is not None

 
# main class

class CALM(Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return x
