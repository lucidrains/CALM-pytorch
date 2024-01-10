from math import ceil

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from beartype import beartype
from beartype.typing import List, Optional

from einops import rearrange, repeat

from x_transformers.x_transformers import (
    RMSNorm,
    Attention,
    TransformerWrapper,
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

# function for returning an ordered list of modules, where the output of the module is the output of that transformer block layer
# ex. for x-transformers TransformerWrapper

@beartype
def transformer_blocks(transformer: Module) -> List[Module]:
    blocks = []
    for layer in transformer.attn_layers.layers:
        blocks.append(layer[-1])
    return blocks

# main class

class CALM(Module):
    @beartype
    def __init__(
        self,
        dim_anchor: int,
        dim_augment: int,
        anchor_llm: Module,
        augment_llm: Module,
        augment_every_num_layers = 4,  # in the paper, they do 4
        attn_kwargs: dict = dict()
    ):
        super().__init__()

        # main contribution of paper
        # is showing that both anchor and augment can be frozen, and that cross attention from anchor -> augment every few layers outperforms lora

        freeze_all_layers_(anchor_llm)
        freeze_all_layers_(augment_llm)

        # matching up blocks from anchor to augment LLM, accounting for potential differences in depth

        anchor_transformer_blocks = transformer_blocks(anchor_llm)
        augment_transformer_blocks = transformer_blocks(augment_llm)

        num_anchor_blocks = len(anchor_transformer_blocks)
        num_augment_blocks = len(augment_transformer_blocks)

        num_attended_augment_hiddens = ceil(num_augment_blocks / augment_every_num_layers)
        num_cross_attending_anchor_blocks = min(num_attended_augment_hiddens, num_anchor_blocks)
        anchor_every_num_layers = num_anchor_blocks // num_cross_attending_anchor_blocks

        augment_blocks_to_hook = augment_transformer_blocks[::augment_every_num_layers]
        anchor_blocks_to_hook = anchor_transformer_blocks[::anchor_every_num_layers]

        # number of cross attention

        num_cross_attns = min(len(augment_blocks_to_hook), len(anchor_blocks_to_hook))

        # instantiate cross attentions

        self.cross_attns = ModuleList([
            Attention(dim = dim_anchor, dim_context = dim_augment, **attn_kwargs) for _ in range(num_cross_attns)
        ])

    def parameters(self):
        return self.cross_attns.parameters()

    def forward(
        self,
        x: Tensor
    ):
        return x
