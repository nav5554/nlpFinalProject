# When to Remove the Mask

COMS 4705 (NLP, Fall 2026) final project: Arnav Sehgal & Tobias Thurston.

We study when a masked diffusion language model can safely commit several
predicted tokens at once, and whether a lightweight learned policy (a small MLP
using confidence, stability, and span-structure features) can make better
commit decisions than standard confidence-based decoding.

## Setup

Works on Apple Silicon (MPS), NVIDIA GPUs (CUDA, e.g. Colab/GCP), or CPU.

```bash
git clone <repo-url>
cd <repo>
python3 -m venv .venv && source .venv/bin/activate   # skip on Colab
pip install -r requirements.txt

# Download the model weights (680 MB) into mdlm-local/
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('kuleshov-group/mdlm-no_flashattn-fp32-owt', 'model.safetensors', local_dir='mdlm-local')"

# Check everything works
python setup.py          # one sentence, two masked words
python sanity_check.py   # WikiText-2: one-shot vs. confidence decoding
```

`sanity_check.py` downloads WikiText-2 on first run and caches the tokenized
test split in `data/`.

## Model

We use **MDLM-OWT** (~170M parameters, GPT-2 tokenizer, `[MASK]` id 50257),
kept frozen. `mdlm-local/` contains the model code from
[`kuleshov-group/mdlm-no_flashattn-fp32-owt`](https://huggingface.co/kuleshov-group/mdlm-no_flashattn-fp32-owt)
with these modifications so it runs without flash-attention (including on Macs):

- Replaced `flash_attn`'s rotary embedding with an equivalent pure-PyTorch version
  (`qkv * cos + rotate_half(qkv) * sin`) and removed the `flash_attn` imports.
- Removed TorchScript JIT fusion flags and `@torch.jit.script` decorators
  (unsupported on Apple MPS); inlined `modulate_fused` to preserve the original math.
- Added `self.post_init()` to `MDLM.__init__` for compatibility with transformers v5.

## Credits

MDLM: Sahoo et al., *Simple and Effective Masked Diffusion Language Models*,
NeurIPS 2024 ([code](https://github.com/kuleshov-group/mdlm), Apache-2.0).
