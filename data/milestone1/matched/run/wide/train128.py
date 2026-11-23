"""
Train the learned commit policies on rows from collect_rows.py.

  - Logistic regression (all features)  -> policies/lr.pt
  - MLP (2 x 64 ReLU, sigmoid output)    -> policies/mlp.pt

Both predict P(the model's current guess at a masked position is correct). Both files store
the feature mean/std used for standardization, so sanity_check.py can score with them directly.

Needs: data/rows_train.parquet, data/rows_validation.parquet
Run:   python train_policies.py
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from features import FEATURES, CommitMLP

EPOCHS = 8
BATCH = 4096
LR = 1e-3
HIDDEN = 128
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")

train = pd.read_parquet("data/rows_train.parquet")
val = pd.read_parquet("data/rows_validation.parquet")
print(f"train rows: {len(train):,}   val rows: {len(val):,}")

Xtr = torch.tensor(train[FEATURES].values, dtype=torch.float32)
ytr = torch.tensor(train["label"].values, dtype=torch.float32)
Xva = torch.tensor(val[FEATURES].values, dtype=torch.float32)
yva = val["label"].values

mean = Xtr.mean(0)
std = Xtr.std(0).clamp(min=1e-6)
os.makedirs("policies", exist_ok=True)


def within_step_auroc(df, scores):
    """AUROC within each (policy, window, step), averaged: what top-k commitment depends on."""
    df = df.assign(_score=scores)
    aucs = [roc_auc_score(g["label"], g["_score"])
            for _, g in df.groupby(["policy", "window", "step"])
            if g["label"].nunique() == 2]
    return sum(aucs) / len(aucs)


def report(name, scores):
    print(f"  {name:<22} pooled AUROC {roc_auc_score(yva, scores):.4f}   "
          f"within-step AUROC {within_step_auroc(val, scores):.4f}")


# ---------------------------------------------------------------- logistic regression
print("\n[logistic regression]")
Ztr = ((Xtr - mean) / std).numpy()
lr = LogisticRegression(max_iter=1000).fit(Ztr, ytr.numpy())
coef = torch.tensor(lr.coef_[0], dtype=torch.float32)
intercept = torch.tensor(float(lr.intercept_[0]))
lr_val = (((Xva - mean) / std) @ coef + intercept).numpy()
torch.save({"kind": "lr", "features": FEATURES, "mean": mean, "std": std,
            "coef": coef, "intercept": intercept}, "policies/lr.pt")
print("  saved policies/lr.pt")

# ---------------------------------------------------------------- MLP
print("\n[MLP]")
mlp = CommitMLP(len(FEATURES), HIDDEN).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()
Ztr_t = ((Xtr - mean) / std).to(device)
ytr_t = ytr.to(device)
Zva_t = ((Xva - mean) / std).to(device)

best_auc, best_state = -1.0, None
for epoch in range(EPOCHS):
    mlp.train()
    perm = torch.randperm(len(Ztr_t), device=device)
    total = 0.0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        loss = loss_fn(mlp(Ztr_t[idx]), ytr_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    mlp.eval()
    with torch.no_grad():
        val_logits = mlp(Zva_t).cpu().numpy()
    auc = roc_auc_score(yva, val_logits)
    print(f"  epoch {epoch + 1}: train loss {total / len(perm):.4f}   val pooled AUROC {auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}

mlp.load_state_dict(best_state)
torch.save({"kind": "mlp", "features": FEATURES, "mean": mean, "std": std,
            "hidden": HIDDEN, "state_dict": best_state}, "policies/mlp.pt")
print(f"  saved policies/mlp.pt (best val pooled AUROC {best_auc:.4f})")
with torch.no_grad():
    mlp_val = mlp(Zva_t).cpu().numpy()

# ---------------------------------------------------------------- comparison
print("\n=== validation AUROC ===")
report("raw confidence", val["conf"].values)
report("logistic regression", lr_val)
report("MLP", mlp_val)
print("\nNext: python sanity_check.py  (decodes the test split with the 'lr' and 'mlp' policies)")
