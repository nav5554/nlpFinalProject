# When to Remove the Mask

COMS 4705 (NLP, Fall 2026) final project: Arnav Sehgal & Tobias Thurston.

We study when a masked diffusion language model can safely commit several
predicted tokens at once, and whether a lightweight learned policy (a small MLP
using confidence, stability, and span-structure features) can make better
commit decisions than standard confidence-based decoding.

## Milestone 1

**Research question.** When a frozen masked diffusion language model fills in a
whole missing sentence, does using a small neural network to choose which token
guesses to commit first improve masked-token accuracy over confidence-first
decoding at the same number of decoding steps?

On 48 held-out WikiText-2 validation sentences from 17 articles, confidence-first
recovered 78 of 1,991 masked tokens. With the same frozen model, starting masks,
and 32 passes, the sentence-trained MLP with two 128-unit hidden layers recovered
104 (5.22% vs. 3.92%). The paired 95% article-bootstrap interval for the gain was
+0.61 to +2.09 percentage points. Logistic regression recovered 108 tokens, and
one-shot decoding recovered 173. This pilot used one training seed.

- [Editable milestone draft](data/milestone1/matched/NLP%20Milestone%201.docx)
- [Results across all masking settings](data/milestone1/results-overview.txt) and [historical notebook evidence](data/milestone1/historical-evidence.txt)
- [Sentence pilot results](data/milestone1/matched/results.json), [predictions](data/milestone1/matched/predictions.json), and [paired comparisons](data/milestone1/matched/audit.json)
- [Exact reproduction commands for the sentence pilot](data/milestone1/matched/reproduce.txt) and [base environment setup](data/milestone1/reproduce.txt)
- [Download the complete milestone evidence packet](data/milestone1/Milestone%201%20complete%20evidence.zip)

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
