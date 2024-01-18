from math import ceil
from pathlib import Path
from functools import partial
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn, einsum, Tensor

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import List, Optional, Callable, Type, Tuple, Union, Literal

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

from pytorch_custom_utils.accelerate_utils import (
    model_forward_contexts
)

# types

Sequence = Union[Tuple, List]

def SequenceOf(t):
    return Union[Tuple[t, ...], List[t]]

def SingularOrMany(t):
    return Union[t, SequenceOf(t)]

# helpers

def exists(v):
  return v is not None

def default(v, d):
    return v if exists(v) else d

def xnor(x, y):
    return not (x ^ y)

def cast_tuple(t, length = 1):
    return t if is_bearable(t, Sequence) else ((t,) * length)

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
def x_transformer_blocks(transformer: Module) -> List[Module]:
    blocks = []
    for layer in transformer.attn_layers.layers:
        blocks.append(layer[-1])
    return blocks

# helper classes

class Recorder:
    @beartype
    def __init__(
        self,
        forward_hook_get_hidden: Union[
            Literal['output'],
            Literal['input']
        ] = 'output'
    ):
        self.output = None
        self.forward_hook_get_hidden = forward_hook_get_hidden

    def __call__(self, _, inp, output):
        assert not exists(self.output)
        hidden = output if self.forward_hook_get_hidden == 'output' else inp
        self.output = hidden.detach()

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
        forward_hook_get_hidden: Union[
            Literal['output'],
            Literal['input']
        ] = 'output',
        **kwargs
    ):
        super().__init__()
        self.pre_rmsnorm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.recorder = recorder
        self.context_proj = None

        if linear_project_context:
            self.context_proj = nn.Linear(dim_context, dim)
            dim_context = dim

        self.attn = Attention(
            dim = dim,
            dim_context = dim_context,
            zero_init_output = True,
            gate_value_heads = True,
            **kwargs
        )

        self.context_mask = None
        self.forward_hook_get_hidden = forward_hook_get_hidden

    def set_mask(self, mask: Tensor):
        self.context_mask = mask

    def unset_mask(self):
        self.context_mask = None

    def forward(self, _, inp, out):
        x = out if self.forward_hook_get_hidden == 'output' else inp

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
        augment_llm: Union[Module, SequenceOf(Module)],
        *,
        attn_kwargs: dict = dict(
            linear_project_context = True,
            pre_rmsnorm = True,
            flash = True
        ),
        connections: Optional[SingularOrMany(Tuple[Tuple[int, int], ...],)] = None,
        input_shape: SingularOrMany(Optional[Tuple]) = None,
        forward_mask_to_augment_llm_key: SingularOrMany(Optional[str]) = None,   # if set, will forward the prompt_mask to the augment LLM (in case it is an encoder) with this key
        augment_every_num_layers: int = 4, # in the paper, they do 4
        augment_extract_layers_fn: SingularOrMany(Optional[Callable[[Module], List[Module]]]) = None,
        augment_llm_mask_kwarg: SingularOrMany(Optional[str]) = None,
        anchor_extract_layers_fn: Callable[[Module], List[Module]] = None,
        augment_transformer_blocks: Optional[Union[List[List[Module]], List[Module]]] = None,
        anchor_transformer_blocks: Optional[List[Module]] = None,
        forward_hook_get_hidden: SingularOrMany(Union[Literal['input'], Literal['output']]) = 'output',
        anchor_forward_hook_get_hidden: Union[Literal['input'], Literal['output']] = 'output',
        pad_id: int = -1
    ):
        super().__init__()

        # account for single augmentation llm (which is what the paper did)
        # in this repo, generalizing it to multiple augmentation llms

        augment_llms = cast_tuple(augment_llm)

        if exists(augment_transformer_blocks):
            if is_bearable(augment_transformer_blocks, List[Module]):
                augment_transformer_blocks = [augment_transformer_blocks]

        if exists(connections):
            if is_bearable(connections, Tuple[Tuple[int, int], ...]):
                connections = [connections]

        # main contribution of paper
        # is showing that both anchor and augment can be frozen, and that cross attention from anchor -> augment every few layers outperforms lora

        self.anchor_llm = anchor_llm
        self.augment_llms = nn.ModuleList(augment_llms)

        freeze_all_layers_(self.anchor_llm)
        freeze_all_layers_(self.augment_llms)

        num_augment_llms = len(self.augment_llms)

        # the only parameters being learned are a bunch of cross attention layers
        # attending from anchor to augmentation model(s)

        self.cross_attns = nn.ModuleList([])

        # determine the transformer blocks involved
        # first determine if the blocks are already passed in

        assert xnor(exists(augment_transformer_blocks), exists(anchor_transformer_blocks))

        transformer_blocks_passed_in = exists(augment_transformer_blocks) and exists(anchor_transformer_blocks)

        # if list of transformer blocks for anchor and augment LLM not passed in, then derive it

        # match up blocks from anchor to augment LLM, accounting for potential differences in depth

        if not transformer_blocks_passed_in:
            get_anchor_blocks_fn = x_transformer_blocks if isinstance(anchor_llm, TransformerWrapper) else get_anchor_transformer_blocks_fn
            anchor_transformer_blocks = get_anchor_blocks_fn(self.anchor_llm)

            augment_extract_layers_fn = cast_tuple(augment_extract_layers_fn, num_augment_llms)

            augment_transformer_blocks = []

            for augment_llm, extract in zip(self.augment_llms, augment_extract_layers_fn):
                extract = default(extract, x_transformer_blocks if isinstance(augment_llm, TransformerWrapper) else None)
                assert exists(extract)
                augment_transformer_blocks.append(extract(augment_llm))

        # calculation for determining every Nth layer of augmentation layer hiddens is attended to
        # in paper, they did every 4th layer of 1 augmentation llm

        num_anchor_blocks = len(anchor_transformer_blocks)
        num_augment_blocks = [len(block) for block in augment_transformer_blocks]

        assert num_anchor_blocks > 0 and all([n > 0 for n in num_augment_blocks]), 'no layers found in either anchor or augment attention networks'

        if not exists(connections):
            connections = []

            for one_augment_transformer_blocks, one_num_augment_blocks in zip(augment_transformer_blocks, num_augment_blocks):

                num_attended_augment_hiddens = ceil(one_num_augment_blocks / augment_every_num_layers)
                num_cross_attending_anchor_blocks = min(num_attended_augment_hiddens, num_anchor_blocks)
                anchor_every_num_layers = num_anchor_blocks // num_cross_attending_anchor_blocks

                anchor_layer_indices = [*range(0, len(anchor_transformer_blocks), anchor_every_num_layers)]
                augment_layer_indices = [*range(0, len(one_augment_transformer_blocks), augment_every_num_layers)]

                connections.append(tuple(zip(anchor_layer_indices, augment_layer_indices)))

        assert len(connections) == num_augment_llms

        self.connections = connections

        # from connections, get all paired transformer blocks between anchor and augments

        anchor_to_augment_blocks = []

        for connection, one_augment_transformer_blocks, one_num_augment_blocks in zip(connections, augment_transformer_blocks, num_augment_blocks):

            anchor_layer_indices, augment_layer_indices = tuple(zip(*connection))

            assert all([1 <= i <= len(anchor_transformer_blocks) for i in anchor_layer_indices]), 'you specified anchor llm layers outside of actual number of layers'
            assert all([1 <= i <= len(one_augment_transformer_blocks) for i in augment_layer_indices]), 'you specified augment llm layers outside of actual number of layers'

            anchor_blocks_to_hook = [anchor_transformer_blocks[i - 1] for i in anchor_layer_indices]
            augment_blocks_to_hook = [one_augment_transformer_blocks[i - 1] for i in augment_layer_indices]

            anchor_to_augment_blocks.append((anchor_blocks_to_hook, augment_blocks_to_hook))

        # for deriving hidden dimensions magically

        input_shape = cast_tuple(input_shape, num_augment_llms)
        forward_hook_get_hidden = cast_tuple(forward_hook_get_hidden, num_augment_llms)

        # cross attend from anchor to augment llm using module forward hooks

        all_anchor_dims = []
        all_augment_dims = []

        for (anchor_blocks_to_hook, augment_blocks_to_hook), augment_llm, position, maybe_one_input_shape in zip(anchor_to_augment_blocks, self.augment_llms, forward_hook_get_hidden, input_shape):

            # number of cross attention for one augmentation llm

            num_cross_attns = min(len(augment_blocks_to_hook), len(anchor_blocks_to_hook))

            # use forward hook to automatically figure out model dimensions for augment and anchor models

            anchor_dims = []
            augment_dims = []

            temp_hooks = []

            def get_shape(shapes_arr, _, inp, out):
                hiddens = out if position == 'output' else inp
                shapes_arr.append(hiddens.shape[-1])

            get_anchor_dims = partial(get_shape, anchor_dims)
            get_augment_dims = partial(get_shape, augment_dims)

            for anchor_block, augment_block in zip(anchor_blocks_to_hook, augment_blocks_to_hook):
                temp_hooks.append(anchor_block.register_forward_hook(get_anchor_dims))
                temp_hooks.append(augment_block.register_forward_hook(get_augment_dims))

            default_dummy_input = torch.ones((1, 1), dtype = torch.long)

            if exists(maybe_one_input_shape):
                augment_dummy_input = torch.randn((1, *maybe_one_input_shape))
            else:
                augment_dummy_input = default_dummy_input

            self.anchor_llm(default_dummy_input)
            augment_llm(augment_dummy_input)

            # unregister temporary hooks

            for hook in temp_hooks:
                hook.remove()

            all_anchor_dims.append(anchor_dims)
            all_augment_dims.append(augment_dims)

        # instantiate cross attentions

        for anchor_dims, augment_dims, (anchor_blocks_to_hook, augment_blocks_to_hook), augment_llm, position in zip(all_anchor_dims, all_augment_dims, anchor_to_augment_blocks, self.augment_llms, forward_hook_get_hidden):

            recorders = []
            one_augment_llm_cross_attns = ModuleList([])

            for dim_anchor, dim_augment, _ in zip(anchor_dims, augment_dims, range(num_cross_attns)):
                recorder = Recorder(forward_hook_get_hidden = position)
                recorders.append(recorder)
                one_augment_llm_cross_attns.append(CrossAttentionBlock(dim = dim_anchor, dim_context = dim_augment, recorder = recorder, forward_hook_get_hidden = anchor_forward_hook_get_hidden, **attn_kwargs))

            # connect the two models

            for anchor_block, recorder, cross_attn, augment_block in zip(anchor_blocks_to_hook, recorders, one_augment_llm_cross_attns, augment_blocks_to_hook):
                augment_block.register_forward_hook(recorder)
                anchor_block.register_forward_hook(cross_attn)

            # add to cross_attns

            self.cross_attns.append(one_augment_llm_cross_attns)

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

    @beartype
    def forward(
        self,
        seq: Tensor,
        *,
        prompt: Union[Tensor, SequenceOf(Tensor)],
        prompt_mask: Optional[SingularOrMany(SequenceOf(Tensor))] = None,
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

        # if only one prompt is given with multiple augmentation llms, then just feed that one prompt into all augment llm

        num_augment_llms = len(self.augment_llms)

        prompts = cast_tuple(prompt, num_augment_llms)

        assert len(prompts) == num_augment_llms

        # prompt masks

        if not exists(prompt_mask):
            prompt_mask = tuple((p != self.pad_id if not torch.is_floating_point(p) else None) for p in prompts)

        prompt_mask = cast_tuple(prompt_mask, num_augment_llms)

        prompt_masks = prompt_mask # at this point, should be plural

        assert len(prompt_masks) == num_augment_llms

        # invoke the augment llm, gathering up the hidden states with the forward hook

        with torch.no_grad():

            self.augment_llms.eval()

            for augment_llm, prompt, prompt_mask in zip(self.augment_llms, prompts, prompt_masks):
                augment_llm_kwarg = dict()

                if exists(self.forward_mask_to_augment_llm_key):
                    augment_llm_kwarg = {self.forward_mask_to_augment_llm_key: prompt_mask}

                augment_llm(prompt, **augment_llm_kwarg)

        # set the context mask for the cross attention

        for one_cross_attn in self.cross_attns:
            for cross_attn in one_cross_attn:
                cross_attn.set_mask(prompt_mask)

        # then invoke the anchor llm, which should take care of the cross attending to the augmented llm hidden states

        logits = self.anchor_llm(seq)

        assert logits.ndim == 3, 'anchor llm should return logits in the shape (batch, seq, num tokens)'

        # unset the context mask

        for one_cross_attn in self.cross_attns:
            for cross_attn in one_cross_attn:
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
        max_grad_norm = 0.5,
        grad_accum_steps = 1
    ):
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
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
        self.grad_accum_steps = grad_accum_steps

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

            for context in model_forward_contexts(
                model = self.model,
                accelerator = self.accelerator,
                grad_accum_steps = self.grad_accum_steps
            ):
                with context():
                    data = next(dl_iter)

                    if not isinstance(data, dict):
                        data = dict(zip(self.data_kwarg_names, data))

                    loss = self.model(**data, **forward_kwargs)

                    self.accelerator.backward(loss / self.grad_accum_steps)

            self.print(f'{step + 1}: {loss.item():.3f}')

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
