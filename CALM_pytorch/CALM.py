from math import ceil
from pathlib import Path
from functools import partial
from contextlib import nullcontext, contextmanager

from dataclasses import dataclass

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

from CALM_pytorch.sampling_utils import (
    sample,
    top_p, top_k
)

# types

Sequence = Union[Tuple, List]

HiddenPosition = Union[Literal['input'], Literal['output']]

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

def get_block_output_from_hook_outputs(
    hidden_position: HiddenPosition,
    _, inp, out
):
    maybe_tensor = out if hidden_position == 'output' else inp

    if isinstance(maybe_tensor, tuple):
        maybe_tensor = maybe_tensor[0]

    assert torch.is_tensor(maybe_tensor)
    return maybe_tensor

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
def x_transformer_blocks(transformer: TransformerWrapper) -> List[Module]:
    blocks = []
    for layer in transformer.attn_layers.layers:
        blocks.append(layer[-1])
    return blocks[1::2]

# helper classes

class Recorder:
    @beartype
    def __init__(
        self,
        outputs: Optional[List] = None,
        forward_hook_get_hidden: HiddenPosition = 'output'
    ):
        self.output = default(outputs, [])
        self.get_output_fn = partial(get_block_output_from_hook_outputs, forward_hook_get_hidden)

    def __call__(self, *args):
        hidden = self.get_output_fn(*args)
        self.output.append(hidden.detach())

class ExtractHiddensWrapper(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        blocks: List[Module],
        hidden_positions: SingularOrMany(HiddenPosition) = 'output'
    ):
        super().__init__()
        hidden_positions = cast_tuple(hidden_positions, len(blocks))
        assert len(hidden_positions) == len(blocks)

        self.model = model

        self.outputs = []
        self.recorders = []

        for block, hidden_position in zip(blocks, hidden_positions):
            recorder = Recorder(self.outputs, hidden_position)
            self.recorders.append(recorder)
            block.register_forward_hook(recorder)

    def forward(self, *args, **kwargs):
        self.model(*args, **kwargs)

        outputs = self.outputs.copy()
        self.outputs.clear()
        return outputs

# cross attention wrapper class

class CrossAttentionBlock(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_context,
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

        self.context_proj = None

        self.dim = dim
        self.dim_context = dim_context

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

        self.context = None
        self.context_mask = None
        self.forward_hook_get_hidden = forward_hook_get_hidden

    def set_mask(self, mask: Tensor):
        self.context_mask = mask

    def unset_mask(self):
        self.context_mask = None

    def forward(self, *hook_args):
        x = get_block_output_from_hook_outputs(self.forward_hook_get_hidden, *hook_args)

        context = self.context
        assert exists(context)

        maybe_enable_grad = torch.enable_grad if self.training else nullcontext

        with maybe_enable_grad():
            res = x
            x = self.pre_rmsnorm(x)

            if exists(self.context_proj):
                context = self.context_proj(context)

            out = self.attn(x, context, context_mask = self.context_mask) + res

        return out

# main class

@dataclass
class AugmentParams:
    model: Module
    hidden_position: SingularOrMany(HiddenPosition) = 'output'
    transformer_blocks: Optional[List[Module]] = None
    extract_blocks_fn: Optional[Callable[[Module], List[Module]]] = None
    model_return_hiddens: bool = False
    input_shape: Optional[Tuple[int, ...]] = None
    connections: Optional[Tuple[Tuple[int, int], ...]] = None
    connect_every_num_layers: int = 4 # in the paper, they do 4
    mask_kwarg: Optional[str] = None

class CALM(Module):
    @beartype
    def __init__(
        self,
        anchor_llm: Module,
        augment_llms: SingularOrMany(AugmentParams),
        *,
        attn_kwargs: dict = dict(
            linear_project_context = True,
            pre_rmsnorm = True,
            flash = True
        ),
        anchor_extract_blocks_fn: Callable[[Module], List[Module]] = None,
        anchor_transformer_blocks: Optional[List[Module]] = None,
        anchor_hidden_position: SingularOrMany(HiddenPosition) = 'output',
        pad_id: int = -1
    ):
        super().__init__()

        if isinstance(augment_llms, AugmentParams):
            augment_llms = [augment_llms]

        augment_llms_params = augment_llms

        self.anchor_llm = anchor_llm
        self.augment_llms = nn.ModuleList([])

        # the only parameters being learned are a bunch of cross attention layers
        # attending from anchor to augmentation model(s)

        self.cross_attns = nn.ModuleList([])

        # determine the transformer blocks involved
        # derive the blocks from the model and extraction function, if not

        if not exists(anchor_transformer_blocks):
            get_anchor_blocks_fn = x_transformer_blocks if isinstance(anchor_llm, TransformerWrapper) else anchor_extract_blocks_fn
            anchor_transformer_blocks = get_anchor_blocks_fn(self.anchor_llm)

        anchor_hidden_position = cast_tuple(anchor_hidden_position, len(anchor_transformer_blocks))
        assert len(anchor_transformer_blocks) == len(anchor_hidden_position)

        # wrap each augment llm with a wrapper that extracts the hiddens
        # if the augment llm is already modified to return a List[Tensor], set model_return_hiddens = True

        wrapped_anchor_llm = ExtractHiddensWrapper(
            anchor_llm,
            anchor_transformer_blocks,
            anchor_hidden_position
        )

        for params in augment_llms_params:

            if params.model_return_hiddens:
                self.augment_llms.append(params.model)
                continue

            if not exists(params.transformer_blocks):
                extract = default(params.extract_blocks_fn, x_transformer_blocks if isinstance(params.model, TransformerWrapper) else None)

                assert exists(extract)

                params.transformer_blocks = extract(params.model)

            params.hidden_position = cast_tuple(params.hidden_position, len(params.transformer_blocks))

            assert len(params.hidden_position) == len(params.transformer_blocks)

            wrapped_augment_llm = ExtractHiddensWrapper(
                params.model,
                params.transformer_blocks,
                params.hidden_position
            )

            self.augment_llms.append(wrapped_augment_llm)

        # main contribution of paper
        # is showing that both anchor and augment can be frozen, and that cross attention from anchor -> augment every few layers outperforms lora

        freeze_all_layers_(self.anchor_llm)
        freeze_all_layers_(self.augment_llms)

        num_augment_llms = len(self.augment_llms)

        # extract all forward outputs from all transformer blocks
        # for sanitizing the input (making sure transformer blocks are ordered by execution)
        # and for magically determining hidden dimensions for cross attention

        default_transformer_input = torch.ones((1, 1), dtype = torch.long)

        recording_inputs = [default_transformer_input]

        for params in augment_llms_params:
            maybe_input_shape = params.input_shape

            if exists(maybe_input_shape):
                inp = torch.randn((1, *maybe_input_shape))
            else:
                inp = default_transformer_input

            recording_inputs.append(inp)

        all_blocks = [anchor_transformer_blocks, *[params.transformer_blocks for params in augment_llms_params]]
        all_models = [wrapped_anchor_llm, *self.augment_llms]

        all_outputs = [model(recording_input) for model, recording_input in zip(all_models, recording_inputs)]

        num_anchor_blocks, *num_augment_blocks = [*map(len, all_outputs)]

        assert num_anchor_blocks > 0 and all([n > 0 for n in num_augment_blocks]), 'no layers found in either anchor or augment attention networks'

        anchor_outputs, *augments_outputs = all_outputs

        # calculation for determining every Nth layer of augmentation layer hiddens is attended to
        # in paper, they did every 4th layer of 1 augmentation llm

        for params, augment_outputs in zip(augment_llms_params, augments_outputs):
            if exists(params.connections):
                continue

            one_num_augment_blocks = len(augment_outputs)

            num_attended_augment_hiddens = ceil(one_num_augment_blocks / params.connect_every_num_layers)
            num_cross_attending_anchor_blocks = min(num_attended_augment_hiddens, num_anchor_blocks)
            anchor_every_num_layers = num_anchor_blocks // num_cross_attending_anchor_blocks

            # using 1 indexed, to save on confusion when manually defining connection layer
            # (some researchers will probably not understand 0th layer == 1)

            anchor_layer_indices = [*range(1, len(anchor_outputs) + 1, anchor_every_num_layers)]
            augment_layer_indices = [*range(1, len(augment_outputs) + 1, params.connect_every_num_layers)]

            params.connections = tuple(zip(augment_layer_indices, anchor_layer_indices))

        self.connections = [params.connections for params in augment_llms_params]

        # from connections, get all paired transformer blocks between anchor and augments

        anchor_to_augment_outputs = []

        for connection, params, augment_outputs in zip(self.connections, augment_llms_params, augments_outputs):
            one_num_augment_blocks = len(augment_outputs)

            augment_layer_indices, anchor_layer_indices = tuple(zip(*connection))

            assert all([1 <= i <= len(anchor_outputs) for i in anchor_layer_indices]), 'you specified anchor llm layers outside of actual number of layers'
            assert all([1 <= i <= len(augment_outputs) for i in augment_layer_indices]), 'you specified augment llm layers outside of actual number of layers'

            anchor_blocks = [anchor_transformer_blocks[i - 1] for i in anchor_layer_indices]
            one_anchor_outputs = [anchor_outputs[i - 1] for i in anchor_layer_indices]
            one_anchor_positions = [anchor_hidden_position[i - 1] for i in anchor_layer_indices]
            one_augment_outputs = [augment_outputs[i - 1] for i in augment_layer_indices]

            num_cross_attns = min(len(one_anchor_outputs), len(one_augment_outputs))

            # get anchor dims

            anchor_dims = [one_anchor_output.shape[-1] for one_anchor_output in one_anchor_outputs]
            augment_dims = [one_augment_output.shape[-1] for one_augment_output in one_augment_outputs]

            # cross attentions for one augment llm

            one_augment_llm_cross_attns = ModuleList([])

            for dim_anchor, dim_augment, anchor_position in zip(anchor_dims, augment_dims, one_anchor_positions):
                one_augment_llm_cross_attns.append(CrossAttentionBlock(dim = dim_anchor, dim_context = dim_augment, forward_hook_get_hidden = anchor_position, **attn_kwargs))

            for anchor_block, cross_attn in zip(anchor_blocks, one_augment_llm_cross_attns):
                anchor_block.register_forward_hook(cross_attn)

            self.cross_attns.append(one_augment_llm_cross_attns)

        # cross entropy loss related

        self.pad_id = pad_id

        # forwarding a mask to augment llm

        self.augment_llms_params = augment_llms_params

    def state_dict(self):
        return self.cross_attns.state_dict()

    def load_state_dict(self, pkg, strict = False):
        self.cross_attns.load_state_dict(pkg, strict = strict)

    def parameters(self):
        return self.cross_attns.parameters()

    def release_cross_attn_contexts(self):
        for one_augment_cross_attns in self.cross_attns:
            for cross_attn in one_augment_cross_attns:
                cross_attn.context = None

    def forward_augments(
        self,
        prompt: Tensor,
        prompt_mask: Optional[SingularOrMany(SequenceOf(Tensor))] = None
    ):
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

        augments_hiddens = []

        with torch.no_grad():

            self.augment_llms.eval()

            for augment_llm, params, prompt, prompt_mask in zip(self.augment_llms, self.augment_llms_params, prompts, prompt_masks):
                augment_llm_kwarg = dict()

                if exists(params.mask_kwarg):
                    augment_llm_kwarg = {params.mask_kwarg: prompt_mask}

                one_augment_hiddens = augment_llm(prompt, **augment_llm_kwarg)

                augments_hiddens.append(one_augment_hiddens)

        # set the contexts for each cross attention block for anchor forward

        for one_augment_hiddens, one_augment_cross_attns, one_augment_connections in zip(augments_hiddens, self.cross_attns, self.connections):

            for (augment_layer_index, _), cross_attn in zip(one_augment_connections, one_augment_cross_attns):
            
                cross_attn.context = one_augment_hiddens[augment_layer_index - 1]

        return prompts, prompt_masks

    @contextmanager
    def set_cross_attn_masks(self, masks):
        # set the context mask for the cross attention

        for one_cross_attn, mask in zip(self.cross_attns, masks):
            for cross_attn in one_cross_attn:
                cross_attn.set_mask(mask)

        yield

        # unset the context mask

        for one_cross_attn in self.cross_attns:
            for cross_attn in one_cross_attn:
                cross_attn.unset_mask()


    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        seq_len: int,
        prompt_mask: Optional[SingularOrMany(SequenceOf(Tensor))] = None,
        filter_fn: Callable = top_p,
        filter_kwargs: dict = dict(
            thres = 0.9
        )
    ):
        batch, device = prompt.shape[0], next(self.cross_attns.parameters()).device

        self.eval()

        # run forward on all the augmentation models and collect hidden states

        prompts, prompt_masks = self.forward_augments(prompt = prompt, prompt_mask = prompt_mask)

        with self.set_cross_attn_masks(prompt_masks):

            # sample

            generated =  sample(
                self.anchor_llm,
                prompt,
                seq_len = seq_len,
                filter_fn = filter_fn,
                filter_kwargs = filter_kwargs
            )

            self.release_cross_attn_contexts()

        return generated

    @beartype
    def forward(
        self,
        seq: Tensor,
        *,
        prompt: SingularOrMany(Tensor),
        prompt_mask: Optional[SingularOrMany(Tensor)] = None,
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

        # run forward on all the augmentation models and collect hidden states

        prompts, prompt_masks = self.forward_augments(prompt = prompt, prompt_mask = prompt_mask)

        with self.set_cross_attn_masks(prompt_masks):
            # invoke the anchor llm, which should take care of the cross attending to the augmented llm hidden states

            logits = self.anchor_llm(seq)

            self.release_cross_attn_contexts()

            assert logits.ndim == 3, 'anchor llm should return logits in the shape (batch, seq, num tokens)'

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
