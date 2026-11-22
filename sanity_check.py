"""
Sanity check for MDLM-OWT on WikiText-2.

1. Loads the frozen MDLM model from ./mdlm-local (patched to run on Mac).
2. Takes windows of WikiText-2 test text, randomly masks 15% of tokens.
3. Fills them two ways:
     - one-shot: every mask filled from a single forward pass
     - iterative confidence decoding: commit the most confident guesses over K steps
4. Prints accuracy and one example so you can eyeball the fills.

Run:  python sanity_check.py
"""

import os
import random

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU fallback for any op MPS lacks

import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "./mdlm-local"   # your locally patched copy (Mac-compatible)
MASK_ID = 50257          # GPT-2 vocab is 50257 tokens; MDLM adds [MASK] as id 50257
WINDOW = 256             # tokens of context per example (model was trained on 1024)
N_EXAMPLES = 20
MASK_RATIO = 0.5
STEPS = 8                # decoding steps for iterative confidence decoding
SEED = 0

random.seed(SEED)
torch.manual_seed(SEED)

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# ---------------------------------------------------------------- model
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = model.to(device).eval()
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")


@torch.no_grad()
def predict(x):
    """One forward pass -> (confidence, argmax) per position. x: [B, L] token ids."""
    t = torch.zeros(x.shape[0], device=x.device)       # time conditioning is off; pass zeros
    logits = model(input_ids=x, timesteps=t)
    logits = logits.logits if hasattr(logits, "logits") else logits
    logits = logits.float()
    logits[..., MASK_ID] = -float("inf")               # never predict [MASK] itself
    probs = logits.softmax(-1)
    conf, guess = probs.max(-1)
    return conf, guess


# ---------------------------------------------------------------- data
CACHE = "data/wikitext2_test_ids.pt"
if os.path.exists(CACHE):
    ids = torch.load(CACHE)                       # fast path: tokenized ids saved on disk
else:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    lines = [l for l in ds["text"] if l.strip() and not l.strip().startswith("=")]  # drop blanks + headings
    ids = tok(" ".join(lines), return_tensors="pt").input_ids[0]
    os.makedirs("data", exist_ok=True)
    torch.save(ids, CACHE)
    print(f"saved {len(ids)} tokens to {CACHE}")

starts = random.sample(range(0, len(ids) - WINDOW), N_EXAMPLES)
targets = torch.stack([ids[s:s + WINDOW] for s in starts]).to(device)   # [N, WINDOW]

mask = torch.rand(targets.shape, device=device) < MASK_RATIO
x_masked = targets.clone()
x_masked[mask] = MASK_ID
print(f"masked tokens: {mask.sum().item()} across {N_EXAMPLES} windows of {WINDOW}")

# ---------------------------------------------------------------- one-shot
_, guess = predict(x_masked)
one_shot_acc = (guess[mask] == targets[mask]).float().mean().item()

# ---------------------------------------------------------------- iterative confidence
x = x_masked.clone()
for step in range(STEPS):
    still = x == MASK_ID
    if not still.any():
        break
    conf, guess = predict(x)
    conf = conf.masked_fill(~still, -1.0)              # only consider masked positions
    for b in range(x.shape[0]):
        n_left = still[b].sum().item()
        if n_left == 0:
            continue
        k = -(-n_left // (STEPS - step))               # ceil: spread remaining evenly over steps
        top = conf[b].topk(k).indices
        x[b, top] = guess[b, top]
iter_acc = (x[mask] == targets[mask]).float().mean().item()

print(f"\none-shot accuracy:                 {one_shot_acc:.3f}")
print(f"confidence decoding ({STEPS} steps):  {iter_acc:.3f}")

# ---------------------------------------------------------------- eyeball a few examples
N_SHOW = 30           # how many examples to print
show = slice(0, 80)      # first 80 tokens of each
for b in range(min(N_SHOW, x.shape[0])):
    print(f"\n================ example {b} ================")
    print("--- masked ---")
    print(" ".join("[M]" if mm else tok.decode([t]).strip()
                   for t, mm in zip(targets[b, show].tolist(), mask[b, show].tolist())))
    print("\n--- filled (iterative) ---")
    print(tok.decode(x[b, show].tolist()))
    print("\n--- original ---")
    print(tok.decode(targets[b, show].tolist()))
