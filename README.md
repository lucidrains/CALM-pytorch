<img src="./calm.png" width=400px/>

## CALM - Pytorch

Implementation of CALM from the paper <a href="https://arxiv.org/abs/2401.02412">LLM Augmented LLMs: Expanding Capabilities through Composition</a>, out of Google Deepmind

Can support any number of augmentation LLMs

## Install

```bash
$ pip install CALM-pytorch
```

## Appreciation

- <a href="https://a16z.com/supporting-the-open-source-ai-community/">A16Z Open Source AI Grant Program</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for the generous sponsorships, as well as my other sponsors, for affording me the independence to open source current artificial intelligence research

## Usage

ex. with `x-transformers`

```python
import torch
from x_transformers import TransformerWrapper, Decoder

augment_llm = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 12,
        heads = 8
    )
)

anchor_llm = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 2,
        heads = 8
    )
)

# import CALM wrapper

from CALM_pytorch import CALM, AugmentParams

calm = CALM(
    anchor_llm,
    augment_llms = AugmentParams(
        model = augment_llm,
        connect_every_num_layers = 4
    )
)

# mock input

seq = torch.randint(0, 20000, (1, 1024))
mask = torch.ones((1, 1024)).bool()
prompt = torch.randint(0, 20000, (1, 256))

# forward for finetuning loss

loss = calm(
    seq,
    mask = mask,
    prompt = prompt
)

loss.backward()

# after much training, prompt the composed model

generated = calm.generate(
    prompt = seq[:, :1],
    seq_len = 1024
)

```

To use a handy trainer class using 🤗 Accelerate, just import `FineTuner` and use as follows

```python
trainer = FineTuner(
    calm = calm,
    dataset = dataset,   # returns a dictionary of input kwargs to calm - dict(seq: Tensor, mask: Tensor, prompt: Tensor). it can also return a Tuple, in which data_kwargs needs to be set to the correct ordered value of kwarg names
    batch_size = 16,
    num_train_steps = 10000,
    learning_rate = 3e-4,
    weight_decay = 1e-2,
    warmup_steps = 1000,
    checkpoint_every = 1000
)

trainer()

# checkpoints of the cross attention parameters will be saved to ./checkpoints every 1000 steps
```

To explore multiple augmentation LLMs, simply pass in a list for `augment_llm`

ex.

```python
calm = CALM(
    anchor_llm = anchor_llm,
    augment_llm = [AugmentParams(augment_llm1), AugmentParams(augment_llm2)] # pass in a list of AugmentParams wrapping model and other hparams specific to that transformer
)
```

Say you want to explore different types of connectivity between anchor and augmentation model(s), just pass in the connections as a tuple of tuple integer pairs, specifying the anchor to augment layer number.

```python
calm = CALM(
    anchor_llm = anchor_llm,
    augment_llms = (
        AugmentParams(
            model = augment_llm1,
            connections = (
                (1, 12),  # 1st layer of augment llm1 attended to by 12th layer of anchor llm
                (2, 12),
                (3, 12),
                (4, 12),
            ),
        ),
        AugmentParams(
            model = augment_llm2,
            connections = (
                (6, 1), # 6th layer of augment llm2 attended to by 1st layer of anchor llm
                (6, 2),
                (12, 12),
            )
        )
    )
)
```

CALM setup with 2 specialized augmentation LLMs + a vision transformer

```python
import torch

# pip install vit-pytorch x-transformers

from vit_pytorch.vit import ViT, Attention
from x_transformers import TransformerWrapper, Encoder, Decoder

anchor_llm = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 16,
        dim_head = 2,
        depth = 12,
        heads = 8
    )
)

augment_llm1 = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 16,
        dim_head = 2,
        depth = 12,
        heads = 8
    )
)

augment_llm2 = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 16,
        dim_head = 2,
        depth = 12,
        heads = 8
    )
)

vit = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 256,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

# calm

from CALM_pytorch import CALM, AugmentParams, FineTuner

calm = CALM(
    anchor_llm = anchor_llm,
    augment_llms = (
        AugmentParams(
            model = augment_llm1,
            mask_kwarg = 'mask'
        ),
        AugmentParams(
            model = augment_llm2,
            mask_kwarg = 'mask'
        ),
        AugmentParams(
            model = vit,
            input_shape = (3, 256, 256),
            hidden_position = 'input',
            extract_blocks_fn = lambda vit: [m for m in vit.modules() if isinstance(m, Attention)]
        )
    ),
    attn_kwargs = dict(
        linear_project_context = True,
        pre_rmsnorm = True,
        flash = True
    )
)

seq = torch.randint(0, 20000, (1, 1024))
mask = torch.ones((1, 1024)).bool()

prompt = (
    torch.randint(0, 20000, (1, 256)),
    torch.randint(0, 20000, (1, 256)),
    torch.randn(1, 3, 256, 256)
)

loss = calm(
    seq,
    mask = mask,
    prompt = prompt
)

loss.backward()
```

## Todo

- [x] figure out how to correctly mask augment llm tokens
- [x] auto-derive model dimensions with dummy input
- [x] take care of finetuning training logic
- [x] show example of manual definitions of custom connectivity between 2+ attention networks
- [x] if anchor and augment transformer block modules are directly passed in (without extraction fn), run a dummy input through both networks and order them correctly using hooks
- [x] fix example for x-transformers, as in x-transformers, depth is actually depth x 2, taking hiddens from after attention and ff
- [x] when finely specifying hidden positions, make sure to reorder it if the transformer blocks themselves were passed in and not ordered to begin with
- [x] extend to a list of augmentation llms
    - [x] full connectivity customization
    - [x] custom number of augmentation layers per augmetation llm
    - [x] make simple vit work
        - [x] refactor so extraction fn, mask kwarg, and other related hparams are grouped together under a dictionary of {[augment_llm_name]: {augment_llm_related_hparams}} - use dataclasses
        - [x] show example
- [x] take care of caching the augment hiddens when sampling. forget about anchor kv cache for now
    - [x] logic for not releasing the saved output from recorder, for inference
    - [x] managing cross attention block state for popping the saved output from the recorder
    - [x] move the augmentation forwards into one shared method, and craft out sampling method for anchor

- [ ] show an example with giving the LLM ability to hear as well, using <a href="https://github.com/lucidrains/audiolm-pytorch">hubert or wav2vec</a> wrappers
- [ ] handle a wrapper or function that takes in the sequence and prompt length, and auto derives the inputs to CALM
- [ ] add an option for self attention path way with memory tokens attending to hidden states of all augmentation llms, akin to what was done with <a href="https://github.com/lucidrains/zorro-pytorch">Zorro</a>

## Citations

```bibtex
@inproceedings{Bansal2024LLMAL,
  title   = {LLM Augmented LLMs: Expanding Capabilities through Composition},
  author  = {Rachit Bansal and Bidisha Samanta and Siddharth Dalmia and Nitish Gupta and Shikhar Vashishth and Sriram Ganapathy and Abhishek Bapna and Prateek Jain and Partha Pratim Talukdar},
  year    = {2024},
  url     = {https://api.semanticscholar.org/CorpusID:266755751}
}
```
