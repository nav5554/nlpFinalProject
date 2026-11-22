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

from features import FEATURES, StepTracker, compute_features, predict_full

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
def predict(x):
    """(conf, guess, margin, entropy) per position; see features.predict_full."""
    return predict_full(model, x, MASK_ID, CHUNK)


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


# ---------------------------------------------------------------- decode + log
INT_COLS = {"flipped", "first_step", "left_masked", "right_masked"}          # stored as int8
SHORT_COLS = {"steps_stable", "dist_known", "run_len"}                       # stored as int16


def decode_and_log(targets, mask, policy, split, window_idx):
    """Decode with `policy`, logging one row per still-masked position per step."""
    x = targets.clone()
    x[mask] = MASK_ID
    B, L = x.shape
    n_maskable = L - 2 * MARGIN
    tracker = StepTracker()
    win = torch.tensor(window_idx, device=x.device)
    frames = []

    for step in range(STEPS):
        still = x == MASK_ID
        if not still.any():
            break
        conf, guess, margin, ent = predict(x)
        flipped, dconf, first, stable = tracker.update(guess, conf)
        feats = compute_features(conf, margin, ent, still, flipped, dconf, first, stable,
                                 step, STEPS, n_maskable)

        b_idx, p_idx = still.nonzero(as_tuple=True)
        cols = {
            "split": split,
            "policy": policy,
            "window": win[b_idx].cpu().numpy().astype(np.int32),
            "step": np.full(len(b_idx), step, dtype=np.int16),
            "pos": p_idx.cpu().numpy().astype(np.int16),
        }
        for f in FEATURES:
            dtype = np.int8 if f in INT_COLS else np.int16 if f in SHORT_COLS else np.float32
            cols[f] = feats[f][still].detach().cpu().numpy().astype(dtype)
        cols["label"] = (guess == targets)[still].cpu().numpy().astype(np.int8)
        frames.append(pd.DataFrame(cols))

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
