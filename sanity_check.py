"""
Sanity check for MDLM-OWT on WikiText-2.

1. Loads the frozen MDLM model from ./mdlm-local (patched to run on Mac).
2. Takes windows of WikiText-2 test text, randomly masks 15% of tokens.
3. Fills them several ways on the SAME masks:
     - one-shot: every mask filled from a single forward pass
     - iterative decoding over STEPS steps with different commit policies:
       confidence, edges-first (heuristic), oracle (upper bound), and the learned
       policies lr / mlp if policies/lr.pt / policies/mlp.pt exist (train_policies.py)
4. Prints accuracy and examples so you can eyeball the fills.

Run:  python sanity_check.py
"""

import os
import random
import re

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU fallback for any op MPS lacks

import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from features import StepTracker, compute_features, load_scorer, predict_full, stack_features

MODEL_NAME = "./mdlm-local"   # your locally patched copy (Mac-compatible)
MASK_ID = 50257          # GPT-2 vocab is 50257 tokens; MDLM adds [MASK] as id 50257
WINDOW = 256             # tokens of context per example (model was trained on 1024)
N_EXAMPLES = 300
MASK_RATIO = 0.5
MASKING = "span"         # "span" = contiguous masked runs, "random" = independent per-token coin flips
MEAN_SPAN = 5            # average span length in tokens (span masking only)
MARGIN = 16              # first/last MARGIN tokens of each window are never masked (context on both sides)
STEPS = 32              # decoding steps (learned policies were trained on 32-step rollouts)
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


def predict(x):
    """(conf, guess, margin, entropy) per position, in chunks of CHUNK windows; see features.py."""
    return predict_full(model, x, MASK_ID, CHUNK)


# learned commit policies (trained by train_policies.py); skipped if not trained yet
SCORERS = {}
for name in ["lr", "mlp"]:
    path = f"policies/{name}.pt"
    if os.path.exists(path):
        SCORERS[name] = load_scorer(path, device)
print(f"learned policies loaded: {list(SCORERS) or 'none (run train_policies.py)'}")


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
_, guess, _, _ = predict(x_masked)
one_shot_acc = (guess[mask] == targets[mask]).float().mean().item()

# ---------------------------------------------------------------- iterative decoding
def dist_to_known(still):
    """For each position, distance to the nearest NON-masked token (original text or already committed).
    still: [B, L] bool, True where still masked. Returns [B, L] long. Vectorized over the batch."""
    B, L = still.shape
    pos = torch.arange(L, device=still.device).expand(B, L)
    known = ~still
    big = 10 ** 6
    left = torch.where(known, pos, torch.full_like(pos, -big)).cummax(dim=1).values          # nearest known on the left
    right = torch.where(known, pos, torch.full_like(pos, big)).flip(1).cummin(dim=1).values.flip(1)  # nearest known on the right
    return torch.minimum(pos - left, right - pos)


def decode(x_masked, policy):
    """Fill all masks over STEPS steps, committing k tokens per step.
    policy = "confidence": commit the k most confident guesses (the standard baseline)
    policy = "oracle":     commit correct guesses first (uses the answer key -> upper bound)
    policy = "edges":      commit blanks closest to known text first (spans fill from the edges inward),
                           confidence breaks ties  (hand-written heuristic baseline)
    policy = "lr" / "mlp": commit the guesses the learned policy scores as most likely correct"""
    x = x_masked.clone()
    tracker = StepTracker()
    n_maskable = x.shape[1] - 2 * MARGIN
    for step in range(STEPS):
        still = x == MASK_ID
        if not still.any():
            break
        conf, guess, margin, ent = predict(x)
        flipped, dconf, first, stable = tracker.update(guess, conf)
        score = conf.clone()
        if policy == "oracle":
            correct = (guess == targets) & still
            score = score + 10.0 * correct.float()    # any correct guess outranks every wrong one
        elif policy == "edges":
            d = dist_to_known(still).float()
            score = -10.0 * d + conf                  # nearest-to-known first; conf (0..1) only breaks ties
        elif policy in SCORERS:
            feats = compute_features(conf, margin, ent, still, flipped, dconf, first, stable,
                                     step, STEPS, n_maskable)
            score = torch.full_like(conf, -1e9)
            score[still] = SCORERS[policy](stack_features(feats, still)).float()
        score = score.masked_fill(~still, -1e9)       # only consider masked positions (very low so any masked score wins)
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

x_edges = decode(x_masked, "edges")
edges_acc = (x_edges[mask] == targets[mask]).float().mean().item()

x_learned, learned_acc = {}, {}
for name in SCORERS:
    x_learned[name] = decode(x_masked, name)
    learned_acc[name] = (x_learned[name][mask] == targets[mask]).float().mean().item()

print(f"\none-shot accuracy:                 {one_shot_acc:.3f}")
print(f"confidence decoding ({STEPS} steps):  {iter_acc:.3f}")
print(f"edges-first decoding ({STEPS} steps): {edges_acc:.3f}   <- heuristic: fill spans from the edges inward")
for name, acc in learned_acc.items():
    print(f"{name + ' decoding':<20} ({STEPS} steps): {acc:.3f}   <- learned policy")
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
    print("\n--- filled (edges-first) ---")
    print(tok.decode(x_edges[b, show].tolist()))
    for name, xl in x_learned.items():
        print(f"\n--- filled ({name}) ---")
        print(tok.decode(xl[b, show].tolist()))
    print("\n--- filled (oracle) ---")
    print(tok.decode(x_oracle[b, show].tolist()))
    print("\n--- original ---")
    print(tok.decode(targets[b, show].tolist()))
