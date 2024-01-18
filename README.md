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

from CALM_pytorch import CALM

calm = CALM(
    anchor_llm,
    augment_llm,
    augment_every_num_layers = 4
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
    augment_llm = [augment_llm1, augment_llm2] # pass in a list
)
```

Say you want to explore different types of connectivity between anchor and augmentation model(s), just pass in the connections as a tuple of tuple integer pairs, specifying the anchor to augment layer number.

```python
calm = CALM(
    anchor_llm = anchor_llm,
    augment_llm = [augment_llm1, augment_llm2],
    connections = [
        (
            (12, 1),   # 12th layer of anchor attends to 1st layer of augment llm1
            (12, 2),
            (12, 3),
            (12, 4),
        ),
        (
            (1, 6),    # 1st layer of anchor attends to 6th layer of augment 2
            (2, 6),
            (12, 12),
        ),
        # ... and so on, add vision transformers, whatever
    ]
)
```

## Todo

- [x] figure out how to correctly mask augment llm tokens
- [x] auto-derive model dimensions with dummy input
- [x] take care of finetuning training logic
- [x] show example of manual definitions of custom connectivity between 2+ attention networks
- [x] extend to a list of augmentation llms
    - [x] full connectivity customization
    - [x] custom number of augmentation layers per augmetation llm
    - [x] make simple vit work
        - [ ] show example
        - [ ] refactor so extraction fn, mask kwarg, and other related hparams are grouped together under a dictionary of {[augment_llm_name]: {augment_llm_related_hparams}} - use `TypedDict` + beartype for validation
    - [ ] move the hook logic for deriving hidden shapes to pytorch-custom-utils for reuse

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
