# Project Context: "When to Remove the Mask" (COMS 4705 Final Project)

Arnav Sehgal & Tobias Thurston, Columbia COMS 4705 (NLP), Fall 2026.
This file is a handoff summary of everything decided and built so far. Read it before changing code.

---

## 1. Research question

Masked diffusion language models (MDLMs) fill in `[MASK]` tokens over several forward passes ("steps"). Each step, the model predicts every masked position in parallel, and a **commit policy** decides which predictions to lock in; the rest stay masked and are re-predicted next step with more context. Fewer steps = faster, but committing too many (or the wrong) tokens at once hurts quality.

The standard policy is **confidence-based**: commit the predictions with the highest max-probability.

**Our question:** *Where does confidence-based commitment fail, and can a lightweight learned policy (a small MLP) that sees extra signals (stability across steps, span structure) commit tokens better — reaching the same quality in fewer steps, or better quality at the same steps?*

---

## 2. How the idea evolved (TA feedback)

Original proposal: train a small MLP on 4 features (confidence, entropy, relative position, fraction still masked) to decide which tokens to commit, on WikiText-2 with **random** token masking (25/50/75%).

TA feedback:
> "I'm still slightly unclear on what the input to the MLP will be. I'm worried that it will be pretty hard to beat the standard confidence based decoding and the project will end up with relatively few experiments run and a negative (yet difficult to explain) result. I'd encourage you to think of extensions of this project to more specific settings where we're interested in unmasking specific sets that confidence wouldn't be correct for. (Perhaps doing some kind of sentence infill)."

Our analysis / changes:

- **Original features couldn't beat confidence by construction.** "Fraction masked" is constant across positions within a step (can only change *how many* to commit, never *which*). Entropy is nearly a monotone function of confidence. So the learned ranking ≈ confidence + small position tweak.
- **Key failure mode confidence can't see: coupled blanks.** The model predicts each masked position *independently* (each sees neighbors as `[MASK]`). Adjacent blanks can each be confident yet jointly wrong ("He moved to [M] [M]" → "New" + "Angeles"). Random scattered masks rarely create this; **spans** do.
- **New masking settings:**
  1. Random token masking — control.
  2. **Span masking** — contiguous runs (Poisson, mean ~5 tokens) — main setting.
  3. **Sentence infill** — mask one whole sentence (use 128+ token passages) — hardest setting, the TA's suggestion.
- **New MLP features** (per masked position, per step): confidence, top1–top2 margin, *stability* (did argmax flip since last step, steps stable, Δconfidence), *span structure* (length of masked run it's in, distance to nearest real token, left/right neighbor masked, neighbor confidences), *within-step relative* (confidence rank, confidence − step mean). ~12 features.
- **Label:** 1 if the model's current argmax at that position equals the ground-truth token, else 0 (from the answer key, not from any policy — so the MLP learns where the model is *wrong*, not imitating a policy). Optional upgrade: "correct AND masked neighbors end up correct".
- **Decoding with the MLP:** commit every masked position with MLP score > τ (at least 1 per step; force-fill on last step). Sweep τ.
- **Evaluation:** main plot = accuracy vs. average forward passes (sweep τ), per masking type. Also fixed budgets 2/4/8/16/32 steps.
- **Baselines:** random order; confidence top-k fixed schedule; **confidence thresholding** (main comparison — also adaptive); hand rules (stability-gated, no-adjacent-commits); **oracle** (commits correct guesses first using the answer key → upper bound).
- **Go/no-go checks before training the MLP:**
  - **Oracle gap** (oracle − confidence): is there headroom?
  - **AUROC check:** logistic regression predicting correctness from confidence-only vs. all features, on cached rollout rows. If AUROC jumps, features carry signal.
- **Ablations:** confidence only → + stability → + span → all features.

Framing for the report: "where does confidence fail, and what information fixes it?" — informative whether the MLP wins, ties a heuristic, or loses (oracle gap + AUROC explain why).

---

## 3. Model

- **MDLM-OWT** (Sahoo et al., NeurIPS 2024), ~170M params, 12 blocks, hidden 768, 12 heads, GPT-2 tokenizer, vocab 50258, **`[MASK]` id = 50257**, trained on OpenWebText at length 1024, `time_conditioning: false` (pass zeros for `timesteps`). Kept **frozen**.
- Bidirectional transformer (no causal mask), rotary position embeddings, adaLN conditioning. Output: logits for every position; we read only masked positions.
- The network has **no commit strategy** — that is entirely our decoding loop. Every policy plugs into the same loop on the same frozen model.
- Why not others: BD3-LM decodes left-to-right in blocks (no right context → bad for infill); DUO is uniform-state diffusion (no `[MASK]` tokens). LLaDA-8B is a possible end-of-project stretch (see §8).

### Local patched copy: `mdlm-local/`
Source: HF `kuleshov-group/mdlm-no_flashattn-fp32-owt`. Patches in `mdlm-local/modeling_mdlm_2.py` so it runs without flash-attention (Mac MPS, Colab CUDA, CPU):
1. Removed `flash_attn` imports; replaced rotary with pure PyTorch: `return qkv * cos + rotate_half(qkv) * sin` (same math as flash's non-interleaved rotary; V slot has cos=1, sin=0).
2. Removed TorchScript JIT fusion flags and the three `@torch.jit.script` decorators (TorchScript fusion crashes on MPS).
3. **Inlined `modulate_fused` as `return x * (1 + scale) + shift`.** Important: the file defines `modulate` twice; without jit.script, `modulate_fused` would resolve to the second overload (extra `unsqueeze`) and silently produce wrong shapes/outputs.
4. Added `self.post_init()` at the end of `MDLM.__init__` (required by transformers v5).

Weights (`model.safetensors`, 680 MB) are gitignored; download with:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('kuleshov-group/mdlm-no_flashattn-fp32-owt', 'model.safetensors', local_dir='mdlm-local')"
```

---

## 4. Repo layout & workflow

```
nlpFinalProject/
  README.md              setup instructions + list of model patches + credits
  requirements.txt       torch transformers datasets einops omegaconf huggingface_hub scikit-learn pandas
  .gitignore             .venv/, data/, *.pt, mdlm-local/model.safetensors, *.bak, mdlm-local/.cache/
  setup.py               minimal smoke test: one sentence, two masked words, prints guess + confidence
  sanity_check.py        WikiText-2 experiment: one-shot vs confidence vs oracle decoding
  mdlm-local/            patched model code + config (weights downloaded separately)
  data/                  cached token ids (generated, gitignored)
```

Workflow: edit locally (VS Code, Mac) → push to GitHub → `git pull` on Google Colab (GPU) → run.

Colab setup cells:
```python
!git clone https://github.com/<owner>/nlpFinalProject.git
%cd nlpFinalProject
!pip install -q -r requirements.txt
!python -c "from huggingface_hub import hf_hub_download; hf_hub_download('kuleshov-group/mdlm-no_flashattn-fp32-owt', 'model.safetensors', local_dir='mdlm-local')"
!python sanity_check.py
```

---

## 5. `sanity_check.py` — what it does now

Settings at top: `WINDOW=256`, `N_EXAMPLES=300`, `MASK_RATIO=0.5`, `STEPS` (8/32/128), `SEED=0`, `CHUNK=32`.

1. **Load** patched MDLM from `./mdlm-local`, GPT-2 tokenizer; device = mps / cuda / cpu.
2. **`predict(x)`**: forward pass in chunks of 32 windows; sets `[MASK]` logit to −inf; softmax; returns `(conf, guess)` = (max prob, argmax) per position.
3. **Data**: `Salesforce/wikitext`, `wikitext-2-raw-v1`, **test split**. Drops blank lines and `= Heading =` lines, applies `wikitext_detokenize()` (undoes WikiText's word-tokenized format: ` @-@ `→`-`, ` @,@ `→`,`, ` @.@ `→`.`, `Gore 's`→`Gore's`, no space before punctuation, `( x )`→`(x)`, non-breaking spaces), joins lines with `\n`, tokenizes with GPT-2, caches to `data/wikitext2_test_ids_clean.pt` (266,099 tokens).
4. **Windows**: `N_EXAMPLES` random 256-token windows (random starts — can overlap; TODO make non-overlapping).
5. **Masking**: random — `mask = torch.rand(shape) < MASK_RATIO` (each token independent coin flip). Span/sentence masking **not implemented yet**.
6. **One-shot**: single forward pass, argmax at every mask.
7. **`decode(x_masked, policy)`**: STEPS iterations; each step re-runs `predict` on current `x`, commits `k = ceil(remaining / steps_left)` positions per window by `topk` on a score:
   - `"confidence"`: score = confidence.
   - `"oracle"`: score = confidence + 10·[guess == true token] → correct guesses always committed first.
   Committed tokens never change. The policy is the only thing that differs between methods; the MLP will be another policy here.
8. **Accuracy** = exact token-ID match over **masked positions only**, pooled over all windows: `(x[mask] == targets[mask]).float().mean()`.
9. Prints N examples: masked / confidence fill / oracle fill / original.

---

## 6. Results so far

| Setup | Windows | Masked tokens | One-shot | Confidence | Oracle |
|---|---|---|---|---|---|
| raw WikiText, 15% random, 8 steps (Mac) | 20 | ~770 | 0.561 | 0.577 | — |
| raw WikiText, 50% random, 8 steps | 20 | 2,559 | 0.365 | 0.363 | — |
| raw WikiText, 50% random, 32 steps | 20 | 2,559 | 0.365 | 0.353 | — |
| **cleaned**, 50% random, 128 steps | 300 | 38,423 | 0.412 | 0.409 | — |
| **cleaned**, 50% random, 32 steps | 300 | ~38K | **0.408** | **0.404** | **0.514** |

**Takeaways:**
- **Oracle beats confidence by ~11 points** (51.4 vs 40.4) with the same model, masks, and steps → commit order matters a lot; confidence is a poor judge of which guesses are safe. Real headroom for the MLP even under random masking. Answers the TA's main concern.
- **Confidence ≈ one-shot.** Explanation: per-token exact match rewards one-shot (each blank gets its best marginal guess given only *real* context, which is ~optimal for expected token hits). Iterative decoding conditions on the model's *own* guesses (right or wrong); its benefit is coherence, which exact match barely rewards (e.g., one-shot "New Angeles" can score more token hits than coherent "New York").
- Detokenizing WikiText raised accuracy ~36% → ~41% at 50% masking.
- 20 windows is too noisy (±2–3 pts); use 300+.

---

## 7. Known issues / observations

- **`<|endoftext|>` at position 0**: MDLM training chunks start with `<|endoftext|>` (id 50256); when position 0 is masked, it confidently predicts it. Fix: frame each window as `[50256] + 254 content tokens + [50256]` and never mask the framing tokens.
- **Edge blanks** have one-sided context. Fix: leave first/last ~16 tokens unmasked (`maskable[:, :16] = False` etc.) and/or start windows at paragraph boundaries (after newline tokens).
- **Repetition** ("Beth Beth Israel", "DreamWorks and DreamWorks", "foul foul") appears **even at 128 steps** (~1 token/step) → partly a model weakness, not only parallel commitment. Track a duplicate-neighbor rate across step counts; the *difference* vs. step count is what commit strategy contributes.
- **Whitespace tokens** sometimes predicted (blank-looking fills). Option: don't mask whitespace-only tokens.
- **Exact token match is strict and favors one-shot.** Add sequence-level metrics: span exact match, repetition rate, GPT-2 perplexity of filled passage. Keep per-token correctness as the MLP *label*.
- Windows currently use random (overlapping) starts — switch to non-overlapping: `ids[: n*W].view(n, W)`.
- WikiText-2 size (256-token windows, approx.): test ~1,000 windows, validation ~900, train ~9–10K. Train → MLP rollouts; validation → τ / AUROC; test → final numbers only.

---

## 8. Next steps (in order)

1. Oracle at several step counts (8, 32, 128) → headroom curve.
2. Data fixes: EOT framing, unmasked margins, non-overlapping (paragraph-aligned) windows.
3. **Span masking** (`span_mask(ratio, mean_span=5)`) and **sentence masking**; rerun one-shot / confidence / oracle. Expect a larger oracle gap on spans.
4. Feature logging inside `decode()`: one row per still-masked position per step → (features, label). Track previous-step argmax for stability features.
5. Rollouts on **train split** (mix confidence-order and random-order rollouts, mix budgets) → cache ~1M rows (Parquet). Split by passage, not row.
6. **AUROC check** (sklearn logistic regression: confidence-only vs. all features) on validation rows.
7. Train MLP (2×64 ReLU, sigmoid, BCE, normalized features) on cached rows.
8. Threshold sweeps (MLP vs. confidence-threshold), fixed-budget comparisons, ablations, plots (accuracy vs. forward passes; oracle = ceiling, random = floor), latency incl. MLP overhead, error analysis.
9. Optional stretch: repeat key comparison on **LLaDA-8B-Base** (`GSAI-ML/LLaDA-8B-Base`, mask id 126336, own tokenizer, no timesteps arg, bf16 ~16 GB → A100; re-tokenize WikiText; compare *gaps* not raw accuracy). Expect possible transformers-version issues.

Compute estimate: rollouts ~160K window-passes; full eval grid ~800K; ~5–15 T4 GPU-hours total, much less on A100. Batch windows (chunks of 32–64) to keep the GPU busy.

---

## 9. References

- Sahoo et al., *Simple and Effective Masked Diffusion Language Models*, NeurIPS 2024 — https://github.com/kuleshov-group/mdlm
- Merity et al., *Pointer Sentinel Mixture Models* (WikiText), 2016 — https://arxiv.org/abs/1609.07843
- Proposal lit review: Hong et al. (learned unmasking policies, ICLR 2026); Sun et al. (TraceLock / token-commitment policy, 2026); Sakurai et al. (controlled study of learned unmasking, 2026).
- LLaDA reference sampler (`generate.py`): `low_confidence` top-k remasking, optional semi-autoregressive blocks.
