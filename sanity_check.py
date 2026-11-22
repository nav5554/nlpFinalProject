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
import re

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU fallback for any op MPS lacks

import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "./mdlm-local"   # your locally patched copy (Mac-compatible)
MASK_ID = 50257          # GPT-2 vocab is 50257 tokens; MDLM adds [MASK] as id 50257
WINDOW = 256             # tokens of context per example (model was trained on 1024)
N_EXAMPLES = 300
MASK_RATIO = 0.5
MASKING = "span"         # "span" = contiguous masked runs, "random" = independent per-token coin flips
MEAN_SPAN = 5            # average span length in tokens (span masking only)
MARGIN = 16              # first/last MARGIN tokens of each window are never masked (context on both sides)
STEPS = 128             # decoding steps for iterative confidence decoding
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


CHUNK = 32               # windows per forward pass (keeps memory bounded)


@torch.no_grad()
def predict(x):
    """Forward pass -> (confidence, argmax) per position. x: [B, L] token ids.
    Runs in chunks of CHUNK windows so large batches don't run out of memory."""
    confs, guesses = [], []
    for i in range(0, x.shape[0], CHUNK):
        xb = x[i:i + CHUNK]
        t = torch.zeros(xb.shape[0], device=xb.device)  # time conditioning is off; pass zeros
        logits = model(input_ids=xb, timesteps=t)
        logits = logits.logits if hasattr(logits, "logits") else logits
        logits = logits.float()
        logits[..., MASK_ID] = -float("inf")            # never predict [MASK] itself
        conf, guess = logits.softmax(-1).max(-1)
        confs.append(conf)
        guesses.append(guess)
    return torch.cat(confs), torch.cat(guesses)


# ---------------------------------------------------------------- data
def wikitext_detokenize(s):
    """Undo WikiText's word-tokenized formatting so it reads like normal web text."""
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")   # 1950 @-@ 51 -> 1950-51
    s = re.sub(r" ('s|'re|'ve|'m|'ll|'d|n't)\b", r"\1", s)                 # Gore 's -> Gore's
    s = re.sub(r" ([,.:;!?])", r"\1", s)                                    # word , -> word,
    s = re.sub(r"\( (.*?) \)", r"(\1)", s)                                 # ( c. 275 ) -> (c. 275)
    s = s.replace("\xa0", " ")                                              # non-breaking spaces
    s = re.sub(r" +", " ", s)                                                # collapse repeated spaces
    return s.strip()


CACHE = "data/wikitext2_test_ids_clean.pt"     # new name so old uncleaned cache is never reused
if os.path.exists(CACHE):
    ids = torch.load(CACHE)                       # fast path: tokenized ids saved on disk
else:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    raw = [l for l in ds["text"] if l.strip() and not l.strip().startswith("=")]  # drop blanks + headings
    lines = [wikitext_detokenize(l) for l in raw]
    print("cleaning example:\n  before:", raw[0].strip()[:150], "\n  after: ", lines[0][:150])
    ids = tok("\n".join(lines), return_tensors="pt").input_ids[0]
    os.makedirs("data", exist_ok=True)
    torch.save(ids, CACHE)
    print(f"saved {len(ids)} tokens to {CACHE}")

starts = random.sample(range(0, len(ids) - WINDOW), N_EXAMPLES)
targets = torch.stack([ids[s:s + WINDOW] for s in starts]).to(device)   # [N, WINDOW]

def random_mask(n, length):
    """Each token in the maskable middle region is masked independently with prob MASK_RATIO."""
    m = torch.rand(n, length) < MASK_RATIO
    m[:, :MARGIN] = False
    m[:, length - MARGIN:] = False
    return m


def span_mask(n, length):
    """Mask contiguous spans (lengths ~ Poisson(MEAN_SPAN), min 1) inside the middle region
    until exactly MASK_RATIO of the maskable tokens are masked."""
    lo, hi = MARGIN, length - MARGIN            # maskable region is [lo, hi)
    region = hi - lo
    target = int(MASK_RATIO * region)
    m = torch.zeros(n, length, dtype=torch.bool)
    for b in range(n):
        while m[b].sum() < target:
            L = max(1, int(torch.poisson(torch.tensor(float(MEAN_SPAN))).item()))
            L = min(L, region)
            start = random.randint(lo, hi - L)
            m[b, start:start + L] = True
        idx = m[b].nonzero().flatten()           # trim overshoot so every window has exactly `target` masks
        m[b, idx[target:]] = False
    return m


def run_lengths(m):
    """Lengths of consecutive masked runs (for reporting)."""
    out = []
    for row in m.tolist():
        run = 0
        for v in row + [False]:
            if v:
                run += 1
            elif run:
                out.append(run)
                run = 0
    return out


if MASKING == "span":
    mask = span_mask(*targets.shape).to(device)
elif MASKING == "random":
    mask = random_mask(*targets.shape).to(device)
else:
    raise ValueError(f"unknown MASKING: {MASKING}")
runs = run_lengths(mask.cpu())
print(f"masking: {MASKING}  (ratio {MASK_RATIO}, margin {MARGIN})  "
      f"mean masked run = {sum(runs) / len(runs):.1f} tokens, longest = {max(runs)}")
x_masked = targets.clone()
x_masked[mask] = MASK_ID
print(f"masked tokens: {mask.sum().item()} across {N_EXAMPLES} windows of {WINDOW}")

# ---------------------------------------------------------------- one-shot
_, guess = predict(x_masked)
one_shot_acc = (guess[mask] == targets[mask]).float().mean().item()

# ---------------------------------------------------------------- iterative decoding
def decode(x_masked, policy):
    """Fill all masks over STEPS steps, committing k tokens per step.
    policy = "confidence": commit the k most confident guesses (the standard baseline)
    policy = "oracle":     commit correct guesses first (uses the answer key -> upper bound)"""
    x = x_masked.clone()
    for step in range(STEPS):
        still = x == MASK_ID
        if not still.any():
            break
        conf, guess = predict(x)
        score = conf.clone()
        if policy == "oracle":
            correct = (guess == targets) & still
            score = score + 10.0 * correct.float()    # any correct guess outranks every wrong one
        score = score.masked_fill(~still, -1.0)       # only consider masked positions
        for b in range(x.shape[0]):
            n_left = still[b].sum().item()
            if n_left == 0:
                continue
            k = -(-n_left // (STEPS - step))           # ceil: spread remaining evenly over steps
            top = score[b].topk(k).indices
            x[b, top] = guess[b, top]
    return x


x = decode(x_masked, "confidence")
iter_acc = (x[mask] == targets[mask]).float().mean().item()

x_oracle = decode(x_masked, "oracle")
oracle_acc = (x_oracle[mask] == targets[mask]).float().mean().item()

print(f"\none-shot accuracy:                 {one_shot_acc:.3f}")
print(f"confidence decoding ({STEPS} steps):  {iter_acc:.3f}")
print(f"oracle decoding ({STEPS} steps):      {oracle_acc:.3f}   <- upper bound for any commit policy")

# ---------------------------------------------------------------- eyeball a few examples
N_SHOW = 30           # how many examples to print
show = slice(0, 96)      # first 96 tokens of each (includes the unmasked margin)
for b in range(min(N_SHOW, x.shape[0])):
    print(f"\n================ example {b} ================")
    print("--- masked ---")
    print(" ".join("[M]" if mm else tok.decode([t]).strip()
                   for t, mm in zip(targets[b, show].tolist(), mask[b, show].tolist())))
    print("\n--- filled (confidence) ---")
    print(tok.decode(x[b, show].tolist()))
    print("\n--- filled (oracle) ---")
    print(tok.decode(x_oracle[b, show].tolist()))
    print("\n--- original ---")
    print(tok.decode(targets[b, show].tolist()))
