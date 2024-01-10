import torch
from torch.nn import Module
from torch import nn, einsum, Tensor

from beartype import beartype

from einops import rearrange, repeat

from x_transformers import (
    Attention
)

# helpers

def exists(v):
  return v is not None
 
# freezing llms

@beartype
def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

# main class

class CALM(Module):
    @beartype
    def __init__(
        self,
        anchor_llm: Module,
        augment_llm: Module
    ):
        super().__init__()
        freeze_all_layers_(anchor_llm)
        freeze_all_layers_(augment_llm)

    def forward(
        self,
        x: Tensor
    ):
        return x
