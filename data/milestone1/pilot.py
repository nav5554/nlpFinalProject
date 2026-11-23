"""Bounded single-gap experiment for nlpFinalProject at 1475d64."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset

import collect_rows as repo
from features import StepTracker, compute_features, load_scorer, span_features, stack_features

SEED = 0
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
GAP_START, GAP_LENGTH = 96, 64
TRAIN_COUNT, TUNE_COUNT, EVAL_COUNT = 64, 16, 24
POLICIES = ("confidence", "edges", "lr", "mlp", "oracle")
repo.CHUNK = 8
assert repo.STEPS == 32 and repo.WINDOW == 256


def token_ids(split):
    # Build from the explicitly named, pinned split. Never consult the repo's test cache.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split=split, revision=REVISION)
    assert str(ds.split) == split, (ds.split, split)
    lines = [repo.wikitext_detokenize(line) for line in ds["text"]
             if line.strip() and not line.strip().startswith("=")]
    ids = repo.tok("\n".join(lines), return_tensors="pt").input_ids[0]
    digest = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    print(f"{split}: {len(ds)} source rows, {len(ids)} cleaned tokens, ids SHA256 {digest}")
    return ids


def examples(ids, indices):
    n = len(ids) // repo.WINDOW
    windows = ids[:n * repo.WINDOW].view(n, repo.WINDOW)
    targets = windows[indices].to(repo.device)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[:, GAP_START:GAP_START + GAP_LENGTH] = True
    return targets, mask


def sample_validation(ids):
    order = random.Random(SEED + 1).sample(range(len(ids) // repo.WINDOW),
                                           TUNE_COUNT + EVAL_COUNT)
    return sorted(order[:TUNE_COUNT]), sorted(order[TUNE_COUNT:])


def collect():
    train = token_ids("train")
    val = token_ids("validation")
    train_idx = sorted(random.Random(SEED).sample(range(len(train) // repo.WINDOW), TRAIN_COUNT))
    tune_idx, eval_idx = sample_validation(val)
    assert not set(tune_idx).intersection(eval_idx)
    Path("data").mkdir(exist_ok=True)
    Path("windows.json").write_text(json.dumps({
        "train": train_idx, "validation_for_training_and_AUROC": tune_idx,
        "validation_for_decoding": eval_idx,
    }, indent=2) + "\n")
    # The upstream collector uses the global PyTorch RNG for its random rollout.
    torch.manual_seed(SEED)
    for split, ids, indices in (("train", train, train_idx), ("validation", val, tune_idx)):
        targets, mask = examples(ids, indices)
        frames = []
        for policy in repo.POLICIES:
            frame, acc = repo.decode_and_log(targets, mask, policy, split, indices)
            print(f"{split}/{policy}: {len(indices)} windows, {len(frame)} rows, "
                  f"label rate {frame.label.mean():.5f}, final accuracy {acc:.5f}")
            assert len(frame) == len(indices) * sum(range(2, GAP_LENGTH + 1, 2))
            assert set(frame.step) == set(range(repo.STEPS))
            frames.append(frame)
        repo.pd.concat(frames, ignore_index=True).to_parquet(f"data/rows_{split}.parquet", index=False)


def decode(targets, mask, policy, scorers):
    # Same scoring rules and quota as sanity_check.py, with the shared features.
    x = targets.masked_fill(mask, repo.MASK_ID)
    tracker = StepTracker()
    quotas = []
    for step in range(repo.STEPS):
        still = x == repo.MASK_ID
        assert torch.all(still.sum(dim=1) == GAP_LENGTH - 2 * step)
        conf, guess, margin, ent = repo.predict(x)
        flipped, dconf, first, stable = tracker.update(guess, conf)
        score = conf
        if policy == "edges":
            score = -10 * span_features(still)[0].float() + conf
        elif policy == "oracle":
            score = 10 * ((guess == targets) & still).float() + conf
        elif policy in scorers:
            feats = compute_features(conf, margin, ent, still, flipped, dconf, first, stable,
                                     step, repo.STEPS, repo.WINDOW - 2 * repo.MARGIN)
            score = torch.full_like(conf, -1e9)
            score[still] = scorers[policy](stack_features(feats, still)).float()
        left = GAP_LENGTH - 2 * step
        k = -(-left // (repo.STEPS - step))
        quotas.append(k)
        top = score.masked_fill(~still, -1e9).topk(k, dim=1).indices
        x.scatter_(1, top, guess.gather(1, top))
    assert quotas == [2] * 32
    assert not torch.any(x == repo.MASK_ID)
    assert torch.equal(x[~mask], targets[~mask])
    return x


def evaluate():
    val = token_ids("validation")
    tune_idx, eval_idx = sample_validation(val)
    manifest = json.loads(Path("windows.json").read_text())
    assert eval_idx == manifest["validation_for_decoding"]
    assert tune_idx == manifest["validation_for_training_and_AUROC"]
    assert not set(tune_idx).intersection(eval_idx)
    targets, mask = examples(val, eval_idx)
    scorers = {name: load_scorer(f"policies/{name}.pt", repo.device) for name in ("lr", "mlp")}
    _, guess, _, _ = repo.predict(targets.masked_fill(mask, repo.MASK_ID))
    outputs = {"one_shot": torch.where(mask, guess, targets)}
    for policy in POLICIES:
        print(f"Decoding {policy} on {len(eval_idx)} validation windows")
        outputs[policy] = decode(targets, mask, policy, scorers)
    n_masked = mask.sum().item()
    correct = {name: ((out == targets) & mask).sum(dim=1).cpu().tolist()
               for name, out in outputs.items()}
    report = {
        "dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1",
        "revision": REVISION, "split": "validation", "device": repo.device,
        "seed": SEED, "validation_sampling_seed": SEED + 1, "passages": len(eval_idx),
        "window_tokens": repo.WINDOW, "gap_start_zero_based": GAP_START,
        "gap_tokens": GAP_LENGTH, "masked_tokens": n_masked,
        "passes_per_passage": {p: 1 if p == "one_shot" else repo.STEPS for p in outputs},
        "iterative_commits_per_pass": 2,
        "accuracy": {p: sum(hits) / n_masked for p, hits in correct.items()},
        "correct_tokens": {p: sum(hits) for p, hits in correct.items()},
        "mlp_minus_confidence_pp": 100 * (sum(correct["mlp"]) - sum(correct["confidence"])) / n_masked,
        "mlp_minus_lr_pp": 100 * (sum(correct["mlp"]) - sum(correct["lr"])) / n_masked,
    }
    predictions = []
    gap = slice(GAP_START, GAP_START + GAP_LENGTH)
    for i, index in enumerate(eval_idx):
        reference = targets[i, gap].cpu().tolist()
        fills = {p: out[i, gap].cpu().tolist() for p, out in outputs.items()}
        predictions.append({
            "window": index, "correct_tokens": {p: counts[i] for p, counts in correct.items()},
            "left_context": repo.tok.decode(targets[i, GAP_START - 12:GAP_START].cpu().tolist()),
            "right_context": repo.tok.decode(targets[i, GAP_START + GAP_LENGTH:GAP_START + GAP_LENGTH + 12].cpu().tolist()),
            "reference": repo.tok.decode(reference),
            "reference_ids": reference,
            "fills": {p: repo.tok.decode(ids) for p, ids in fills.items()},
            "fill_ids": fills,
        })
    Path("results.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "evaluate"))
    args = parser.parse_args()
    print(f"seed={SEED}, validation_sample_seed={SEED+1}, steps={repo.STEPS}, "
          f"gap=[{GAP_START},{GAP_START+GAP_LENGTH}), chunk={repo.CHUNK}, "
          f"train={TRAIN_COUNT}, tune={TUNE_COUNT}, evaluation={EVAL_COUNT}")
    (collect if args.stage == "collect" else evaluate)()
