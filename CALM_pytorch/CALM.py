from math import ceil
from contextlib import nullcontext

import torch
import torch.nn.functional as F
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

# helper classes

class Recorder:
    def __init__(self):
        self.output = None

    def __call__(self, _, __, out):
        assert not exists(self.output)
        self.output = out.detach()

    def pop_saved(self):
        output = self.output
        assert exists(output)
        self.output = None
        return output

# cross attention wrapper class

class CrossAttentionBlock(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_context,
        recorder: Recorder,
        linear_project_context = True,  # in the paper, they do a projection on the augmented hidden states. not sure if this is needed though, but better to be accurate first
        pre_rmsnorm = False,
        **kwargs
    ):
        super().__init__()
        self.pre_rmsnorm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.recorder = recorder
        self.context_proj = None

        if linear_project_context:
            self.context_proj = nn.Linear(dim_context, dim)
            dim_context = dim

        self.attn = Attention(dim = dim, dim_context = dim_context, **kwargs)

    def forward(self, _, __, x):

        context = self.recorder.pop_saved()
        maybe_enable_grad = torch.enable_grad if self.training else nullcontext

        with maybe_enable_grad():
            res = x
            x = self.pre_rmsnorm(x)

            if exists(self.context_proj):
                context = self.context_proj(context)

            out = self.attn(x, context) + res

        return out

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
        attn_kwargs: dict = dict(
            linear_project_context = True,
            pre_rmsnorm = True,
            flash = True
        ),
        ignore_index = -1
    ):
        super().__init__()

        # main contribution of paper
        # is showing that both anchor and augment can be frozen, and that cross attention from anchor -> augment every few layers outperforms lora

        freeze_all_layers_(anchor_llm)
        freeze_all_layers_(augment_llm)

        self.anchor_llm = anchor_llm
        self.augment_llm = augment_llm

        # matching up blocks from anchor to augment LLM, accounting for potential differences in depth

        anchor_transformer_blocks = transformer_blocks(anchor_llm)
        augment_transformer_blocks = transformer_blocks(augment_llm)

        num_anchor_blocks = len(anchor_transformer_blocks)
        num_augment_blocks = len(augment_transformer_blocks)

        assert num_anchor_blocks > 0 and num_augment_blocks > 0, 'no layers found in either anchor or augment attention networks'

        num_attended_augment_hiddens = ceil(num_augment_blocks / augment_every_num_layers)
        num_cross_attending_anchor_blocks = min(num_attended_augment_hiddens, num_anchor_blocks)
        anchor_every_num_layers = num_anchor_blocks // num_cross_attending_anchor_blocks

        augment_blocks_to_hook = augment_transformer_blocks[::-1][::augment_every_num_layers][::-1]
        anchor_blocks_to_hook = anchor_transformer_blocks[::anchor_every_num_layers]

        # number of cross attention

        num_cross_attns = min(len(augment_blocks_to_hook), len(anchor_blocks_to_hook))

        # instantiate cross attentions

        self.recorders = []
        self.cross_attns = ModuleList([])

        for _ in range(num_cross_attns):
            recorder = Recorder()
            self.recorders.append(recorder)
            self.cross_attns.append(CrossAttentionBlock(dim = dim_anchor, dim_context = dim_augment, recorder = recorder, **attn_kwargs))

        # connect the two models

        for anchor_block, recorder, cross_attn, augment_block in zip(anchor_blocks_to_hook, self.recorders, self.cross_attns, augment_blocks_to_hook):
            augment_block.register_forward_hook(recorder)
            anchor_block.register_forward_hook(cross_attn)

        # cross entropy loss related

        self.ignore_index = ignore_index

    def parameters(self):
        return self.cross_attns.parameters()

    def forward(
        self,
        x: Tensor,
        return_loss = True
    ):
        if return_loss:
            self.cross_attns.train()
            self.anchor_llm.train()
            x, labels = x[:, :-1], x[:, 1:]

        with torch.no_grad():
            self.augment_llm.eval()
            _ = self.augment_llm(x)

        logits = self.anchor_llm(x)

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss
