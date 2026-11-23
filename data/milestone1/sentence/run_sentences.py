"""Evaluate the saved fixed-gap policies on complete sentences, without refitting."""
import hashlib
import json
import random
import re
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
import torch
from datasets import load_dataset
from nltk.tokenize import PunktTokenizer

import collect_rows as repo
from features import StepTracker, compute_features, load_scorer, span_features, stack_features

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SEED = 43
METHODS = ("confidence", "edges", "lr", "mlp", "oracle")
repo.CHUNK = 8
assert repo.WINDOW == 256 and repo.STEPS == 32


def sample_sentences():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                           split="validation", revision=REVISION)
    assert str(dataset.split) == "validation"
    segmenter = PunktTokenizer("english")
    cleaned, paragraphs, article_edges = [], [], [0]
    cursor = 0
    for row, raw in enumerate(dataset["text"]):
        stripped = raw.strip()
        if re.fullmatch(r"= [^=].*? =", stripped):
            if cursor != article_edges[-1]:
                article_edges.append(cursor)
        if not stripped or stripped.startswith("="):
            continue
        line = repo.wikitext_detokenize(raw)
        article = len(article_edges) - 1
        paragraphs.append((row, article, cursor, line))
        cleaned.append(line)
        cursor += len(line) + 1
    text = "\n".join(cleaned)
    article_edges.append(len(text))
    encoded = repo.tok(text, return_offsets_mapping=True, verbose=False)
    ids = encoded.input_ids
    print("Validation IDs SHA-256:", hashlib.sha256(np.array(ids, dtype=np.int64).tobytes()).hexdigest())
    start_offsets, end_offsets = zip(*encoded.offset_mapping)
    assert list(start_offsets) == sorted(start_offsets) and list(end_offsets) == sorted(end_offsets)
    previous = json.loads((ROOT / "run/windows.json").read_text())
    excluded = set(previous["validation_for_training_and_AUROC"] + previous["validation_for_decoding"])
    candidates = []
    for row, article, offset, line in paragraphs:
        spans = list(segmenter.span_tokenize(line))
        for lo, hi in spans[1:-1]:
            a, b = offset + lo, offset + hi
            if not re.search(r"""[.!?]["'’”)]*$""", text[a:b]):
                continue
            left = bisect_right(end_offsets, a)
            right = bisect_left(start_offsets, b)
            length = right - left
            if not 32 <= length <= 64 or repo.tok.decode(ids[left:right]).strip() != text[a:b].strip():
                continue
            window = left - (repo.WINDOW - length) // 2
            end = window + repo.WINDOW
            if window < 0 or end > len(ids):
                continue
            if start_offsets[window] < article_edges[article] or end_offsets[end - 1] > article_edges[article + 1]:
                continue
            if any(i in excluded for i in range(window // 256, (end - 1) // 256 + 1)):
                continue
            candidates.append((article, row, window, left, right, a, b))
    random.Random(SEED).shuffle(candidates)
    selected, used, per_article = [], [], defaultdict(int)
    for article, row, window, left, right, a, b in candidates:
        if per_article[article] >= 2 or any(abs(window - other) < 256 for other in used):
            continue
        selected.append(dict(article=article, source_row=row, window_start=window,
                             gap_start=left-window, gap_length=right-left, sentence=text[a:b]))
        used.append(window)
        per_article[article] += 1
        if len(selected) == 48:
            break
    assert len(selected) >= 24, len(selected)
    selected.sort(key=lambda entry: entry["window_start"])
    print(f"{len(candidates)} candidates, {len(selected)} sampled from {len(per_article)} articles")
    (OUT / "sample.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False)+"\n")
    targets = torch.tensor([ids[p["window_start"]:p["window_start"]+256] for p in selected], device=repo.device)
    pos = torch.arange(256, device=repo.device)
    start = torch.tensor([p["gap_start"] for p in selected], device=repo.device)
    length = torch.tensor([p["gap_length"] for p in selected], device=repo.device)
    mask = (pos >= start[:, None]) & (pos < (start+length)[:, None])
    return selected, targets, mask


def decode(targets, mask, policy, scorers):
    x = targets.masked_fill(mask, repo.MASK_ID)
    remaining = mask.sum(dim=1)
    tracker = StepTracker()
    for step in range(32):
        still = x == repo.MASK_ID
        assert torch.equal(still.sum(1), remaining)
        conf, guess, margin, ent = repo.predict(x)
        flipped, dconf, first, stable = tracker.update(guess, conf)
        score = conf
        if policy == "edges":
            score = -10 * span_features(still)[0].float() + conf
        elif policy == "oracle":
            score = 10 * ((guess == targets) & still).float() + conf
        elif policy in scorers:
            feats = compute_features(conf, margin, ent, still, flipped, dconf, first, stable,
                                     step, 32, 256 - 2 * repo.MARGIN)
            score = torch.full_like(conf, -1e9)
            score[still] = scorers[policy](stack_features(feats, still)).float()
        k = -(-remaining // (32-step))
        assert torch.all((k >= 1) & (k <= 2))
        top = score.masked_fill(~still, -1e9).topk(2, dim=1).indices
        replacements = torch.where(
            torch.arange(2, device=x.device) < k[:, None], guess.gather(1, top), x.gather(1, top))
        x.scatter_(1, top, replacements)
        remaining = remaining - k
    assert not torch.any(x == repo.MASK_ID)
    assert torch.equal(x[~mask], targets[~mask])
    return x


def comparison(rows, baseline):
    deltas = [p["correct"]["mlp"]-p["correct"][baseline] for p in rows]
    by_article = defaultdict(lambda: [0, 0])
    for p, delta in zip(rows, deltas, strict=True):
        sums = by_article[p["article"]]
        sums[0] += delta
        sums[1] += p["gap_length"]
    pairs = np.array(list(by_article.values()))
    choice = np.random.default_rng(SEED).integers(len(pairs), size=(10000, len(pairs)))
    boot = pairs[choice].sum(axis=1)
    return dict(delta_pp=100*sum(deltas)/sum(p["gap_length"] for p in rows),
                wins=sum(d>0 for d in deltas), ties=sum(d==0 for d in deltas),
                losses=sum(d<0 for d in deltas),
                article_bootstrap_95_pp=np.quantile(100*boot[:,0]/boot[:,1],[.025,.975]).tolist())


print("Seed:", SEED, "dataset revision:", REVISION, "Punkt NLTK version:", nltk.__version__)
for name in ("lr", "mlp"):
    path = ROOT / f"run/policies/{name}.pt"
    print(name, hashlib.sha256(path.read_bytes()).hexdigest())
rows, targets, mask = sample_sentences()
if "--sample-only" in sys.argv:
    raise SystemExit(0)
scorers = {name: load_scorer(ROOT / f"run/policies/{name}.pt", repo.device) for name in ("lr","mlp")}
_, guess, _, _ = repo.predict(targets.masked_fill(mask, repo.MASK_ID))
outputs = {"one_shot": torch.where(mask, guess, targets)}
for method in METHODS:
    print("Decoding", method)
    outputs[method] = decode(targets, mask, method, scorers)
for i, p in enumerate(rows):
    masked_ids = targets[i,mask[i]].cpu().tolist()
    p["reference_ids"] = masked_ids
    p["fill_ids"] = {method: out[i,mask[i]].cpu().tolist() for method,out in outputs.items()}
    p["fills"] = {method: repo.tok.decode(ids) for method,ids in p["fill_ids"].items()}
    p["correct"] = {method: sum(a==b for a,b in zip(masked_ids,ids,strict=True)) for method,ids in p["fill_ids"].items()}
total = sum(p["gap_length"] for p in rows)
hits = {method: sum(p["correct"][method] for p in rows) for method in outputs}
lengths = [p["gap_length"] for p in rows]
result = dict(windows=len(rows), articles=len({p["article"] for p in rows}), masked_tokens=total,
              gap_min=min(lengths), gap_median=float(np.median(lengths)), gap_max=max(lengths),
              seed=SEED, iterative_passes=32, one_shot_passes=1,
              correct=hits, accuracy={m:hits[m]/total for m in hits},
              fully_recovered={m:sum(p["correct"][m]==p["gap_length"] for p in rows) for m in hits},
              comparisons={m:comparison(rows,m) for m in ("confidence","lr","one_shot")})
(OUT / "predictions.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False)+"\n")
(OUT / "results.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
