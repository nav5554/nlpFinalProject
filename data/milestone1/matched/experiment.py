"""Sentence-matched training and ranker-width comparison at 32 passes."""
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
SEED = 53
METHODS = ("confidence", "edges", "lr", "mlp", "oracle")
repo.CHUNK = 8
assert repo.WINDOW == 256 and repo.STEPS == 32


def sample_sentences(split, role, count):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                           split=split, revision=REVISION)
    assert str(dataset.split) == split
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
    print(split, "IDs SHA-256:", hashlib.sha256(np.array(ids, dtype=np.int64).tobytes()).hexdigest())
    start_offsets, end_offsets = zip(*encoded.offset_mapping)
    assert list(start_offsets) == sorted(start_offsets) and list(end_offsets) == sorted(end_offsets)
    old = json.loads((ROOT / "run/windows.json").read_text())
    previous_sentences = json.loads((ROOT / "sentence/sample.json").read_text())
    previous_articles = {s["article"] for s in previous_sentences}
    if split == "validation":
        for i in old["validation_for_training_and_AUROC"] + old["validation_for_decoding"]:
            first = bisect_right(article_edges, start_offsets[256*i]) - 1
            last = bisect_right(article_edges, end_offsets[256*i+255] - 1) - 1
            previous_articles.update(range(first, last+1))
    candidates = []
    for row, article, offset, line in paragraphs:
        if role == "evaluation" and article in previous_articles:
            continue
        if role == "tuning" and article not in previous_articles:
            continue
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
            candidates.append((article, row, window, left, right, a, b))
    random.Random(SEED).shuffle(candidates)
    selected, used, per_article = [], set(), defaultdict(int)
    # Prefer distinct articles before including a second window from any article.
    for cap in range(1, 5):
        for article, row, window, left, right, a, b in candidates:
            if per_article[article] >= cap or any(abs(window - other) < 256 for other in used):
                continue
            selected.append(dict(article=article, source_row=row, window_start=window,
                                 gap_start=left-window, gap_length=right-left, sentence=text[a:b]))
            used.add(window)
            per_article[article] += 1
            if len(selected) == count:
                break
        if len(selected) == count:
            break
    assert len(selected) >= 4
    selected.sort(key=lambda entry: entry["window_start"])
    print(role, split, "windows", len(selected), "articles", len(set(x["article"] for x in selected)),
          "eligible candidates", len(candidates))
    (OUT / f"{role}_sample.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False)+"\n")
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



def collect():
    import pandas as pd

    for split, role, count in [("train","training",64), ("validation","tuning",16)]:
        records, targets, mask = sample_sentences(split, role, count)
        torch.manual_seed(0)
        frames = []
        for policy in ("confidence","random"):
            print("Collecting", split, policy, flush=True)
            frame, _ = repo.decode_and_log(targets, mask, policy, split, [r["window_start"] for r in records])
            assert set(frame.step) == set(range(32))
            frames.append(frame)
        table = pd.concat(frames, ignore_index=True)
        table.to_parquet(OUT / f"run/data/rows_{split}.parquet", index=False)
        print(split, "rows", len(table))


@torch.no_grad()
def initial_candidates(targets, mask):
    """Diagnostic on the initial context. It is independent of the decoding budget."""
    hits = {1:0, 5:0, 10:0}
    for i in range(0, len(targets), repo.CHUNK):
        ref = targets[i:i+repo.CHUNK]
        gaps = mask[i:i+repo.CHUNK]
        x = ref.masked_fill(gaps, repo.MASK_ID)
        output = repo.model(input_ids=x, timesteps=torch.zeros(len(x),device=repo.device))
        logits = (output.logits if hasattr(output, "logits") else output)[gaps].float()
        logits[:,repo.MASK_ID] = -float("inf")
        top = logits.topk(10, dim=-1).indices
        for k in hits:
            hits[k] += (top[:,:k] == ref[gaps][:,None]).any(1).sum().item()
    return hits


def evaluate():
    rows, targets, mask = sample_sentences("validation","evaluation",48)
    tune = json.loads((OUT / "tuning_sample.json").read_text())
    assert not {r["article"] for r in tune} & {r["article"] for r in rows}
    scorers = {name: load_scorer(path,repo.device) for name,path in {
        "mlp":OUT/"run/policies/mlp.pt", "lr":OUT/"run/policies/lr.pt",
        "wide":OUT/"run/policies/wide.pt", "prior_mlp":ROOT/"run/policies/mlp.pt",
        "prior_lr":ROOT/"run/policies/lr.pt"}.items()}
    candidate_hits = initial_candidates(targets,mask)
    _, guess, _, _ = repo.predict(targets.masked_fill(mask, repo.MASK_ID))
    outputs = {"one_shot": torch.where(mask, guess, targets)}
    for method in ["confidence","edges","lr","mlp","wide","prior_lr","prior_mlp","oracle"]:
        print("Evaluating",method,flush=True)
        outputs[method]=decode(targets,mask,method,scorers)
    for i,r in enumerate(rows):
        ref=targets[i,mask[i]].cpu().tolist()
        r["reference_ids"]=ref
        r["fill_ids"]={m:x[i,mask[i]].cpu().tolist() for m,x in outputs.items()}
        r["correct"]={m:sum(a==b for a,b in zip(ref,ids,strict=True)) for m,ids in r["fill_ids"].items()}
    (OUT/"predictions.json").write_text(json.dumps(rows,indent=2)+"\n")
    hits={m:sum(r["correct"][m] for r in rows) for m in outputs}
    tokens=sum(r["gap_length"] for r in rows)
    report={"articles":len({r["article"] for r in rows}), "passages":len(rows),"masked_tokens":tokens,
            "hits":hits,"accuracy":{m:n/tokens for m,n in hits.items()},
            "comparisons_to_mlp64":{m:comparison(rows,m) for m in outputs if m!="mlp"},
            "initial_topk_hits":candidate_hits, "base_params":sum(p.numel() for p in repo.model.parameters())}
    (OUT/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect()
    elif sys.argv[1] == "evaluate":
        evaluate()
    elif sys.argv[1] == "preview":
        sample_sentences("validation","evaluation",48)
