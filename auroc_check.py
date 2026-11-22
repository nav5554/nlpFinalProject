"""
AUROC check: do features beyond confidence help predict whether a guess is correct?

Fits logistic regressions on rows from the train split and scores AUROC on rows from
the validation split (different passages, so no leakage).

Needs: data/rows_train.parquet and data/rows_validation.parquet (run collect_rows.py first).
Run:  python auroc_check.py
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONF = ["conf"]
CONF_PLUS = ["conf", "margin", "entropy"]
STABILITY = ["flipped", "steps_stable", "dconf", "first_step"]
SPAN = ["dist_known", "run_len", "left_masked", "right_masked", "left_conf", "right_conf"]
RELATIVE = ["conf_rank", "conf_minus_mean", "frac_masked", "step_frac"]
ALL = CONF_PLUS + STABILITY + SPAN + RELATIVE

train = pd.read_parquet("data/rows_train.parquet")
val = pd.read_parquet("data/rows_validation.parquet")
print(f"train rows: {len(train):,}  (label rate {train.label.mean():.3f})")
print(f"val rows:   {len(val):,}  (label rate {val.label.mean():.3f})")


def fit(cols, data=train):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(data[cols], data["label"])
    return clf


def auroc(clf, cols, data=val):
    return roc_auc_score(data["label"], clf.predict_proba(data[cols])[:, 1])


# ---------------------------------------------------------------- 1. main comparison
print("\n=== AUROC on validation (higher = better at telling right guesses from wrong) ===")
print(f"  raw confidence (no model)       {roc_auc_score(val.label, val.conf):.4f}")
ladder = [
    ("confidence only", CONF),
    ("+ margin, entropy", CONF_PLUS),
    ("+ stability", CONF_PLUS + STABILITY),
    ("+ span structure", CONF_PLUS + STABILITY + SPAN),
    ("all features", ALL),
]
models = {}
for name, cols in ladder:
    clf = fit(cols)
    models[name] = (clf, cols)
    print(f"  {name:<30}  {auroc(clf, cols):.4f}")

# ---------------------------------------------------------------- 2. single features
print("\n=== single-feature AUROC on validation (0.5 = no signal; direction ignored) ===")
singles = []
for c in ALL:
    a = roc_auc_score(val.label, val[c])
    singles.append((max(a, 1 - a), c, "higher=more likely correct" if a >= 0.5 else "higher=more likely wrong"))
for a, c, d in sorted(singles, reverse=True):
    print(f"  {c:<18} {a:.4f}   ({d})")

# ---------------------------------------------------------------- 3. where does it help?
clf_conf, cols_conf = models["confidence only"]
clf_all, cols_all = models["all features"]
print("\n=== confidence-only vs all-features AUROC by situation (validation) ===")
groups = {
    "edge of span (dist_known == 1)": val.dist_known == 1,
    "inside span (dist_known >= 2)": val.dist_known >= 2,
    "deep inside (dist_known >= 4)": val.dist_known >= 4,
    "confidence rollouts": val.policy == "confidence",
    "random rollouts": val.policy == "random",
    "early steps (step_frac < 0.5)": val.step_frac < 0.5,
    "late steps (step_frac >= 0.5)": val.step_frac >= 0.5,
}
for name, sel in groups.items():
    sub = val[sel]
    if sub.label.nunique() < 2:
        continue
    print(f"  {name:<34} n={len(sub):>8,}  conf {auroc(clf_conf, cols_conf, sub):.4f}"
          f"  all {auroc(clf_all, cols_all, sub):.4f}")

# ---------------------------------------------------------------- 4. what the full model uses
print("\n=== all-features logistic regression coefficients (standardized; sign = direction) ===")
coefs = clf_all.named_steps["logisticregression"].coef_[0]
for c, w in sorted(zip(cols_all, coefs), key=lambda t: -abs(t[1])):
    print(f"  {c:<18} {w:+.3f}")

# ---------------------------------------------------------------- 5. within-step AUROC
# Commit decisions only compare blanks in the SAME window at the SAME step. Pooled AUROC also
# compares rows across steps/windows, so features that are constant within a step
# (frac_masked, step_frac) can raise it without changing which tokens get picked.
# Within-step AUROC = AUROC computed separately per (policy, window, step), then averaged.
def within_step_auroc(df, scores):
    df = df.assign(_score=scores)
    aucs = [roc_auc_score(g["label"], g["_score"])
            for _, g in df.groupby(["policy", "window", "step"])
            if g["label"].nunique() == 2]
    return sum(aucs) / len(aucs), len(aucs)


print("\n=== WITHIN-STEP AUROC on validation (what top-k commitment actually depends on) ===")
base, n_groups = within_step_auroc(val, val["conf"])
print(f"  (averaged over {n_groups:,} window-steps that have both right and wrong guesses)")
print(f"  raw confidence                  {base:.4f}")
for name, (clf, cols) in models.items():
    a, _ = within_step_auroc(val, clf.predict_proba(val[cols])[:, 1])
    print(f"  {name:<30}  {a:.4f}   ({a - base:+.4f} vs confidence)")

print("\n  by rollout policy:")
for pol in ["confidence", "random"]:
    sub = val[val.policy == pol]
    c, _ = within_step_auroc(sub, sub["conf"])
    a, _ = within_step_auroc(sub, clf_all.predict_proba(sub[cols_all])[:, 1])
    print(f"    {pol:>10} rollouts: confidence {c:.4f}   all features {a:.4f}   ({a - c:+.4f})")

print("\nRead: if 'all features' clearly beats 'confidence only' (say +0.02 AUROC or more),"
      "\nthe extra features carry signal confidence misses, so the MLP has something to learn."
      "\nWithin-step AUROC is the stricter test: a gain there means better choice of WHICH tokens"
      "\nto commit; a gain only in pooled AUROC means better calibration across steps (useful for"
      "\ndeciding HOW MANY to commit with a threshold).")
