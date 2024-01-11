<img src="./calm.png" width=400px/>

## CALM - Pytorch

Implementation of CALM from the paper <a href="https://arxiv.org/abs/2401.02412">LLM Augmented LLMs: Expanding Capabilities through Composition</a>, out of Google Deepmind

Will attempt to generalize this to any number of LLMs

<a href="https://discord.gg/Vmm5nrMZWc">Temporary discord discussion</a>

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

## Todo

- [x] figure out how to correctly mask augment llm tokens
- [x] auto-derive model dimensions with dummy input
- [x] take care of finetuning training logic

- [ ] handle a wrapper or function that takes in the sequence and prompt length, and auto derives the inputs to CALM
- [ ] show example of manually passing in list of transformer blocks as `List[Module]`. try out with some popular pretrained models

## Citations

```bibtex
@inproceedings{Bansal2024LLMAL,
  title   = {LLM Augmented LLMs: Expanding Capabilities through Composition},
  author  = {Rachit Bansal and Bidisha Samanta and Siddharth Dalmia and Nitish Gupta and Shikhar Vashishth and Sriram Ganapathy and Abhishek Bapna and Prateek Jain and Partha Pratim Talukdar},
  year    = {2024},
  url     = {https://api.semanticscholar.org/CorpusID:266755751}
}
```
