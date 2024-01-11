from pathlib import Path
from math import ceil
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn, einsum, Tensor

from beartype import beartype
from beartype.typing import List, Optional, Callable, Type, Tuple

from einops import rearrange, repeat

from x_transformers.x_transformers import (
    RMSNorm,
    Attention,
    TransformerWrapper,
)

from accelerate import Accelerator

from pytorch_custom_utils import (
    OptimizerWithWarmupSchedule,
    get_adam_optimizer,
    auto_unwrap_model
)

# helpers

def exists(v):
  return v is not None

def xnor(x, y):
    return not (x ^ y)

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

        self.attn = Attention(dim = dim, dim_context = dim_context, zero_init_output = True, **kwargs)

        self.context_mask = None

    def set_mask(self, mask: Tensor):
        self.context_mask = mask

    def unset_mask(self):
        self.context_mask = None

    def forward(self, _, __, x):

        context = self.recorder.pop_saved()
        maybe_enable_grad = torch.enable_grad if self.training else nullcontext

        with maybe_enable_grad():
            res = x
            x = self.pre_rmsnorm(x)

            if exists(self.context_proj):
                context = self.context_proj(context)

            out = self.attn(x, context, context_mask = self.context_mask) + res

        return out

# main class

class CALM(Module):
    @beartype
    def __init__(
        self,
        anchor_llm: Module,
        augment_llm: Module,
        *,
        attn_kwargs: dict = dict(
            linear_project_context = True,
            pre_rmsnorm = True,
            flash = True
        ),
        forward_mask_to_augment_llm_key: Optional[str] = None,   # if set, will forward the prompt_mask to the augment LLM (in case it is an encoder) with this key
        augment_every_num_layers = 4, # in the paper, they do 4
        get_augment_transformer_blocks_fn: Callable[[Module], List[Module]] = lambda module: module.blocks,
        get_anchor_transformer_blocks_fn: Callable[[Module], List[Module]] = lambda module: module.blocks,
        augment_transformer_blocks: Optional[List[Module]] = None,
        anchor_transformer_blocks: Optional[List[Module]] = None,
        pad_id = -1
    ):
        super().__init__()

        # main contribution of paper
        # is showing that both anchor and augment can be frozen, and that cross attention from anchor -> augment every few layers outperforms lora

        freeze_all_layers_(anchor_llm)
        freeze_all_layers_(augment_llm)

        self.anchor_llm = anchor_llm
        self.augment_llm = augment_llm

        assert xnor(exists(augment_transformer_blocks), exists(anchor_transformer_blocks))

        transformer_blocks_passed_in = exists(augment_transformer_blocks) and exists(anchor_transformer_blocks)

        # if list of transformer blocks for anchor and augment LLM not passed in, then derive it

        # match up blocks from anchor to augment LLM, accounting for potential differences in depth

        if not transformer_blocks_passed_in:
            if isinstance(anchor_llm, TransformerWrapper):
                anchor_transformer_blocks = transformer_blocks(anchor_llm)
            else:
                anchor_transformer_blocks = get_anchor_transformer_blocks_fn(anchor_llm)

            if isinstance(augment_llm, TransformerWrapper):
                augment_transformer_blocks = transformer_blocks(augment_llm)
            else:
                augment_transformer_blocks = get_augment_transformer_blocks_fn(augment_llm)

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

        # use forward hook to automatically figure out model dimensions for augment and anchor models

        anchor_dims = []
        augment_dims = []

        temp_hooks = []
        get_anchor_dims = lambda _, __, out: anchor_dims.append(out.shape[-1])
        get_augment_dims = lambda _, __, out: augment_dims.append(out.shape[-1])

        for anchor_block, augment_block in zip(anchor_blocks_to_hook, augment_blocks_to_hook):
            temp_hooks.append(anchor_block.register_forward_hook(get_anchor_dims))
            temp_hooks.append(augment_block.register_forward_hook(get_augment_dims))

        dummy_input = torch.ones((1, 1), dtype = torch.long)
        self.anchor_llm(dummy_input)
        self.augment_llm(dummy_input)

        # unregister temporary hooks

        for hook in temp_hooks:
            hook.remove()

        # instantiate cross attentions

        self.recorders = []
        self.cross_attns = ModuleList([])

        for dim_anchor, dim_augment, _ in zip(anchor_dims, augment_dims, range(num_cross_attns)):
            recorder = Recorder()
            self.recorders.append(recorder)
            self.cross_attns.append(CrossAttentionBlock(dim = dim_anchor, dim_context = dim_augment, recorder = recorder, **attn_kwargs))

        # connect the two models

        for anchor_block, recorder, cross_attn, augment_block in zip(anchor_blocks_to_hook, self.recorders, self.cross_attns, augment_blocks_to_hook):
            augment_block.register_forward_hook(recorder)
            anchor_block.register_forward_hook(cross_attn)

        # cross entropy loss related

        self.pad_id = pad_id

        # forwarding a mask to augment llm

        self.forward_mask_to_augment_llm_key = forward_mask_to_augment_llm_key

    def state_dict(self):
        return self.cross_attns.state_dict()

    def load_state_dict(self, pkg, strict = False):
        self.cross_attns.load_state_dict(pkg, strict = strict)

    def parameters(self):
        return self.cross_attns.parameters()

    def forward(
        self,
        seq: Tensor,
        *,
        prompt: Tensor,
        mask: Optional[Tensor] = None,
        return_loss = True,
        anchor_llm_in_train_mode = True  # unsure about this
    ):
        if return_loss:
            self.cross_attns.train()

            if anchor_llm_in_train_mode:
                self.anchor_llm.train()
            else:
                self.anchor_llm.eval()

            seq, labels = seq[:, :-1], seq[:, 1:]

        prompt_mask = prompt != self.pad_id

        # invoke the augment llm, gathering up the hidden states with the forward hook

        with torch.no_grad():
            augment_llm_kwarg = dict()

            if exists(self.forward_mask_to_augment_llm_key):
                augment_llm_kwarg = {self.forward_mask_to_augment_llm_key: prompt_mask}

            self.augment_llm.eval()
            _ = self.augment_llm(prompt)

        # set the context mask for the cross attention

        for cross_attn in self.cross_attns:
            cross_attn.set_mask(prompt_mask)

        # then invoke the anchor llm, which should take care of the cross attending to the augmented llm hidden states

        logits = self.anchor_llm(seq)

        assert logits.ndim == 3, 'anchor llm should return logits in the shape (batch, seq, num tokens)'

        # unset the context mask

        for cross_attn in self.cross_attns:
            cross_attn.unset_mask()

        # return logits for decoding

        if not return_loss:
            return logits

        # account for prompt masking

        if exists(mask):
            labels = labels.masked_fill(~mask[:, 1:], self.pad_id)

        # for fine tuning

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.pad_id
        )

        return loss

# fine tune trainer

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

@auto_unwrap_model()
class FineTuner:

    @beartype
    def __init__(
        self,
        calm: CALM,
        *,
        num_train_steps: int,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        dataset: Dataset,
        data_kwarg_names: Tuple[str, ...] = ('seq', 'mask', 'prompt'),
        accelerate_kwargs: dict = dict(),
        checkpoint_every: int = 1000,
        checkpoint_path: str = './checkpoints',
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        warmup_steps: int = 1000,
        max_grad_norm = 0.5
    ):
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        self.data_kwarg_names = data_kwarg_names

        self.model = calm

        adam = get_adam_optimizer(
            calm.parameters(),
            lr = learning_rate,
            wd = weight_decay
        )

        self.optimizer = OptimizerWithWarmupSchedule(
            accelerator = self.accelerator,
            optimizer = adam,
            scheduler = scheduler,
            scheduler_kwargs = scheduler_kwargs,
            warmup_steps = warmup_steps,
            max_grad_norm = max_grad_norm
        )

        self.step = 0
        self.num_train_steps = num_train_steps

        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(exist_ok = True, parents = True)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def print(self, msg):
        self.accelerator.print(msg)

    def save(self, filename: str, overwrite: bool = True):
        path = self.checkpoint_path / filename
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            step = self.step
        )

        torch.save(pkg, str(path))

    def load(self, filename: str):
        path = self.checkpoint_path / filename
        assert path.exists()

        pkg = torch.load(str(path))

        self.model.load_state_dict(pkg['model'])
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.step = pkg['step']

    def __call__(self, forward_kwargs: dict = dict()):
        dl_iter = cycle(self.dl)
        self.model.train()

        for step in range(self.step, self.num_train_steps):
            data = next(dl_iter)

            if not isinstance(data, dict):
                data = dict(zip(self.data_kwarg_names, data))

            loss = self.model(**data, **forward_kwargs)

            self.print(f'{step + 1}: {loss.item():.3f}')
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.step += 1

            self.accelerator.wait_for_everyone()

            if self.is_main and not (self.step % self.checkpoint_every):
                num = self.step // self.checkpoint_every
                self.save(f'checkpoint.{num}.pt')

            self.accelerator.wait_for_everyone()

        self.print('training complete')
        self.save('checkpoint.-1.pt')
