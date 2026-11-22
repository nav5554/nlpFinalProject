"""
Shared code for the learned commit policies.

One source of truth for: the model forward pass (conf / guess / margin / entropy), the
per-position features, the step-to-step stability tracker, and the MLP architecture.
Used by collect_rows.py (to build training data), train_policies.py (to train),
and sanity_check.py (to decode with the trained policies), so features always match.
"""

import torch
import torch.nn as nn

# Feature columns, in the order the policies expect them.
FEATURES = [
    # confidence-type
    "conf", "margin", "entropy",
    # stability across steps
    "flipped", "steps_stable", "dconf", "first_step",
    # span structure
    "dist_known", "run_len", "left_masked", "right_masked", "left_conf", "right_conf",
    # relative to this step
    "conf_rank", "conf_minus_mean", "frac_masked", "step_frac",
]


# ---------------------------------------------------------------- forward pass
@torch.no_grad()
def predict_full(model, x, mask_id, chunk):
    """Forward pass in chunks. Returns per-position (conf, guess, margin, entropy), each [B, L].
    conf = top-1 prob, margin = top-1 minus top-2 prob, entropy of the predictive distribution."""
    confs, guesses, margins, ents = [], [], [], []
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        t = torch.zeros(xb.shape[0], device=xb.device)   # MDLM time conditioning is off
        out = model(input_ids=xb, timesteps=t)
        logits = (out.logits if hasattr(out, "logits") else out).float()
        logits[..., mask_id] = -float("inf")             # never predict [MASK]
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


# ---------------------------------------------------------------- stability tracking
class StepTracker:
    """Tracks, per position, whether the argmax changed since the last step and for how
    many consecutive steps it has stayed the same. Call update() once per step, right after
    the forward pass."""

    def __init__(self):
        self.prev_guess = None
        self.prev_conf = None
        self.stable = None

    def update(self, guess, conf):
        if self.prev_guess is None:
            flipped = torch.zeros_like(guess, dtype=torch.bool)
            dconf = torch.zeros_like(conf)
            first = torch.ones_like(guess, dtype=torch.bool)
            self.stable = torch.zeros_like(guess, dtype=torch.long)
        else:
            flipped = guess != self.prev_guess
            dconf = conf - self.prev_conf
            first = torch.zeros_like(guess, dtype=torch.bool)
            self.stable = torch.where(flipped, torch.zeros_like(self.stable), self.stable + 1)
        self.prev_guess, self.prev_conf = guess, conf
        return flipped, dconf, first, self.stable


# ---------------------------------------------------------------- features
def span_features(still):
    """still: [B, L] True where masked. Returns distance to nearest known token, length of the
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


def compute_features(conf, margin, ent, still, flipped, dconf, first, stable,
                     step, n_steps, n_maskable):
    """All features as a dict of [B, L] tensors (only masked positions are meaningful)."""
    B, L = conf.shape
    dist, run_len, lm, rm = span_features(still)
    lconf = torch.ones_like(conf)
    lconf[:, 1:] = torch.where(still[:, :-1], conf[:, :-1], torch.ones_like(conf[:, :-1]))
    rconf = torch.ones_like(conf)
    rconf[:, :-1] = torch.where(still[:, 1:], conf[:, 1:], torch.ones_like(conf[:, 1:]))
    n_masked = still.sum(1, keepdim=True)
    conf_mean = (conf * still).sum(1, keepdim=True) / n_masked.clamp(min=1)
    return {
        "conf": conf,
        "margin": margin,
        "entropy": ent,
        "flipped": flipped,
        "steps_stable": stable,
        "dconf": dconf,
        "first_step": first,
        "dist_known": dist,
        "run_len": run_len,
        "left_masked": lm,
        "right_masked": rm,
        "left_conf": lconf,
        "right_conf": rconf,
        "conf_rank": conf_rank(conf, still),
        "conf_minus_mean": conf - conf_mean,
        "frac_masked": (n_masked.float() / n_maskable).expand(B, L),
        "step_frac": torch.full_like(conf, step / n_steps),
    }


def stack_features(feats, still):
    """[N, F] float tensor of the masked positions' features, in FEATURES order."""
    return torch.stack([feats[f][still].float() for f in FEATURES], dim=1)


# ---------------------------------------------------------------- policies
class CommitMLP(nn.Module):
    """Predicts P(current guess is correct) from the features. Outputs logits."""

    def __init__(self, n_in=len(FEATURES), hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_scorer(path, device):
    """Load a saved policy (lr.pt or mlp.pt) as a function: [N, F] features -> [N] scores.
    Higher score = more likely correct. Both files store the feature mean/std used in training."""
    ckpt = torch.load(path, map_location=device)
    assert ckpt["features"] == FEATURES, "feature list changed since this policy was trained"
    mean = ckpt["mean"].to(device)
    std = ckpt["std"].to(device)
    if ckpt["kind"] == "lr":
        coef = ckpt["coef"].to(device)
        intercept = ckpt["intercept"].to(device)
        return lambda X: ((X - mean) / std) @ coef + intercept
    mlp = CommitMLP(len(FEATURES), ckpt.get("hidden", 64)).to(device)
    mlp.load_state_dict(ckpt["state_dict"])
    mlp.eval()

    @torch.no_grad()
    def score(X):
        return mlp((X - mean) / std)
    return score
