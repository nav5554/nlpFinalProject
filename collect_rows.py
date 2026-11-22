"""
Collect per-token rows for the AUROC check and (later) MLP training.

For each window: span-mask it, then decode step by step with a rollout policy
("confidence" or "random"). At every step, for every still-masked position, log one
row of features plus the label: is the model's current guess at that position correct?

Outputs (gitignored):
  data/rows_train.parquet       rows from the WikiText-2 train split      (fit models here)
  data/rows_validation.parquet  rows from the WikiText-2 validation split (score AUROC here)

Run:  python collect_rows.py
"""

import os
import random
import re

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU fallback for any op MPS lacks

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ---------------------------------------------------------------- settings
MODEL_NAME = "./mdlm-local"
MASK_ID = 50257
WINDOW = 256
MASK_RATIO = 0.5
MEAN_SPAN = 5            # mean span length (Poisson)
MARGIN = 16              # first/last MARGIN tokens never masked
STEPS = 32               # decoding steps per rollout
CHUNK = 16               # windows per forward pass (lower if out of memory)
N_WINDOWS = {"train": 300, "validation": 100}
POLICIES = ["confidence", "random"]   # rollout policies (mix -> varied states for the MLP)
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device).eval()


# ---------------------------------------------------------------- model
@torch.no_grad()
def predict(x):
    """Forward pass in chunks. Returns per-position (conf, guess, margin, entropy), each [B, L].
    conf = top-1 prob, margin = top-1 minus top-2 prob, entropy of the predictive distribution."""
    confs, guesses, margins, ents = [], [], [], []
    for i in range(0, x.shape[0], CHUNK):
        xb = x[i:i + CHUNK]
        t = torch.zeros(xb.shape[0], device=xb.device)   # time conditioning is off
        out = model(input_ids=xb, timesteps=t)
        logits = (out.logits if hasattr(out, "logits") else out).float()
        logits[..., MASK_ID] = -float("inf")             # never predict [MASK]
        logp = logits.log_softmax(-1)
        del logits
        top2 = logp.topk(2, dim=-1)
        p2 = top2.values.exp()
        confs.append(p2[..., 0])
        margins.append(p2[..., 0] - p2[..., 1])
        guesses.append(top2.indices[..., 0])
        ents.append(-(logp.exp() * logp.clamp(min=-1e4)).sum(-1))   # clamp avoids 0 * -inf
        del logp
    return torch.cat(confs), torch.cat(guesses), torch.cat(margins), torch.cat(ents)


# ---------------------------------------------------------------- data
def wikitext_detokenize(s):
    """Undo WikiText's word-tokenized formatting (same as sanity_check.py)."""
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ('s|'re|'ve|'m|'ll|'d|n't)\b", r"\1", s)
    s = re.sub(r" ([,.:;!?])", r"\1", s)
    s = re.sub(r"\( (.*?) \)", r"(\1)", s)
    s = s.replace("\xa0", " ")
    s = re.sub(r" +", " ", s)
    return s.strip()


def load_split(split):
    """Cleaned, tokenized ids for a WikiText-2 split, cached in data/."""
    cache = f"data/wikitext2_{split}_ids_clean.pt"
    if os.path.exists(cache):
        return torch.load(cache)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    lines = [wikitext_detokenize(l) for l in ds["text"]
             if l.strip() and not l.strip().startswith("=")]
    ids = tok("\n".join(lines), return_tensors="pt").input_ids[0]
    os.makedirs("data", exist_ok=True)
    torch.save(ids, cache)
    print(f"saved {len(ids)} tokens to {cache}")
    return ids


def sample_windows(ids, n):
    """n non-overlapping WINDOW-token windows. Returns (windows [n, WINDOW], window indices)."""
    n_total = len(ids) // WINDOW
    all_windows = ids[: n_total * WINDOW].view(n_total, WINDOW)
    idx = sorted(random.sample(range(n_total), min(n, n_total)))
    return all_windows[idx], idx


def span_mask(n, length):
    """Contiguous spans (Poisson(MEAN_SPAN), min 1) inside [MARGIN, length - MARGIN)
    until exactly MASK_RATIO of that region is masked (same as sanity_check.py)."""
    lo, hi = MARGIN, length - MARGIN
    region = hi - lo
    target = int(MASK_RATIO * region)
    m = torch.zeros(n, length, dtype=torch.bool)
    for b in range(n):
        while m[b].sum() < target:
            L = max(1, int(torch.poisson(torch.tensor(float(MEAN_SPAN))).item()))
            L = min(L, region)
            start = random.randint(lo, hi - L)
            m[b, start:start + L] = True
        idx = m[b].nonzero().flatten()
        m[b, idx[target:]] = False
    return m


# ---------------------------------------------------------------- features
def span_features(still):
    """still: [B, L] True where masked. Returns dist to nearest known token, length of the
    masked run containing each position, and whether the left/right neighbor is masked."""
    B, L = still.shape
    pos = torch.arange(L, device=still.device).expand(B, L)
    known = ~still
    big = 10 ** 6
    left = torch.where(known, pos, torch.full_like(pos, -big)).cummax(dim=1).values
    right = torch.where(known, pos, torch.full_like(pos, big)).flip(1).cummin(dim=1).values.flip(1)
    dist = torch.minimum(pos - left, right - pos)
    run_len = right - left - 1
    left_masked = torch.zeros_like(still)
    left_masked[:, 1:] = still[:, :-1]
    right_masked = torch.zeros_like(still)
    right_masked[:, :-1] = still[:, 1:]
    return dist, run_len, left_masked, right_masked


def conf_rank(conf, still):
    """Within each window, rank of each masked position's confidence among masked positions,
    scaled to [0, 1] (1 = most confident)."""
    B, L = conf.shape
    order = conf.masked_fill(~still, -1.0).argsort(dim=1)       # unmasked sort to the bottom
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(L, device=conf.device).expand(B, L))
    n_masked = still.sum(1, keepdim=True)
    r = (ranks - (L - n_masked)).float()
    return r / (n_masked - 1).clamp(min=1).float()


# ---------------------------------------------------------------- decode + log
def decode_and_log(targets, mask, policy, split, window_idx):
    """Decode with `policy`, logging one row per still-masked position per step."""
    x = targets.clone()
    x[mask] = MASK_ID
    B, L = x.shape
    n_maskable = L - 2 * MARGIN
    prev_guess = prev_conf = None
    stable = torch.zeros(B, L, dtype=torch.long, device=x.device)
    win = torch.tensor(window_idx, device=x.device)
    frames = []

    for step in range(STEPS):
        still = x == MASK_ID
        if not still.any():
            break
        conf, guess, margin, ent = predict(x)

        # stability across steps
        if prev_guess is None:
            flipped = torch.zeros_like(still)
            dconf = torch.zeros_like(conf)
            first = torch.ones_like(still)
        else:
            flipped = guess != prev_guess
            dconf = conf - prev_conf
            first = torch.zeros_like(still)
            stable = torch.where(flipped, torch.zeros_like(stable), stable + 1)

        # span structure + neighbors
        dist, run_len, lm, rm = span_features(still)
        lconf = torch.ones_like(conf)
        lconf[:, 1:] = torch.where(still[:, :-1], conf[:, :-1], torch.ones_like(conf[:, :-1]))
        rconf = torch.ones_like(conf)
        rconf[:, :-1] = torch.where(still[:, 1:], conf[:, 1:], torch.ones_like(conf[:, 1:]))

        # relative-to-this-step
        n_masked = still.sum(1, keepdim=True)
        conf_mean = (conf * still).sum(1, keepdim=True) / n_masked.clamp(min=1)
        rank = conf_rank(conf, still)
        frac = (n_masked.float() / n_maskable).expand(B, L)

        label = guess == targets
        b_idx, p_idx = still.nonzero(as_tuple=True)

        def g(t, dtype=np.float32):
            return t[still].detach().cpu().numpy().astype(dtype)

        frames.append(pd.DataFrame({
            "split": split,
            "policy": policy,
            "window": win[b_idx].cpu().numpy().astype(np.int32),
            "step": np.full(len(b_idx), step, dtype=np.int16),
            "pos": p_idx.cpu().numpy().astype(np.int16),
            # confidence-type
            "conf": g(conf),
            "margin": g(margin),
            "entropy": g(ent),
            # stability
            "flipped": g(flipped, np.int8),
            "steps_stable": g(stable, np.int16),
            "dconf": g(dconf),
            "first_step": g(first, np.int8),
            # span structure
            "dist_known": g(dist, np.int16),
            "run_len": g(run_len, np.int16),
            "left_masked": g(lm, np.int8),
            "right_masked": g(rm, np.int8),
            "left_conf": g(lconf),
            "right_conf": g(rconf),
            # relative to this step
            "conf_rank": g(rank),
            "conf_minus_mean": g(conf - conf_mean),
            "frac_masked": g(frac),
            "step_frac": np.full(len(b_idx), step / STEPS, dtype=np.float32),
            # label
            "label": g(label, np.int8),
        }))

        # commit k tokens per window according to the rollout policy
        if policy == "confidence":
            score = conf.clone()
        elif policy == "random":
            score = torch.rand_like(conf)
        else:
            raise ValueError(policy)
        score = score.masked_fill(~still, -1e9)
        n_left = still.sum(1)
        for b in range(B):
            nl = n_left[b].item()
            if nl == 0:
                continue
            k = -(-nl // (STEPS - step))
            top = score[b].topk(k).indices
            x[b, top] = guess[b, top]

        prev_guess, prev_conf = guess, conf

    acc = (x[mask] == targets[mask]).float().mean().item()
    return pd.concat(frames, ignore_index=True), acc


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for split, n in N_WINDOWS.items():
        ids = load_split(split)
        windows, window_idx = sample_windows(ids, n)
        targets = windows.to(device)
        mask = span_mask(*targets.shape).to(device)
        print(f"\n[{split}] {len(window_idx)} windows, {mask.sum().item()} masked tokens")

        dfs = []
        for policy in POLICIES:
            df, acc = decode_and_log(targets, mask, policy, split, window_idx)
            print(f"  {policy:>10} rollout: final accuracy {acc:.3f}, {len(df):,} rows, "
                  f"label rate {df.label.mean():.3f}")
            dfs.append(df)

        out = pd.concat(dfs, ignore_index=True)
        path = f"data/rows_{split}.parquet"
        out.to_parquet(path, index=False)
        print(f"  saved {len(out):,} rows to {path}")
