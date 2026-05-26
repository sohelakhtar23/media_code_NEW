# MediaEval 2026 — Predicting Commercial Memorability (Subtask 2)

I'm working on **MediaEval 2026 — Predicting Commercial Memorability (Subtask 2)**. The task is to predict two continuous scores for 85 test videos: `memorability_score` and `brand_memorability`. Trained on 339 labeled financial brand videos. Evaluation metric: **Spearman ρ**, independently for each target.

---

## Key Dataset Facts

- All videos are financial institution content (Goldman Sachs, JPMorgan, UBS, etc.) — visually and semantically homogeneous
- Annotators watched **at most 60 seconds** of each video regardless of actual duration
- 32 training channels, 9 test channels — **zero channel overlap**
- GPU: NVIDIA RTX 4000 Ada (20GB VRAM)

---

## Features Extracted

| Feature | Dim | Notes |
|---|---|---|
| Frame stats (OpenCV) | 10d | brightness, saturation, colorfulness, face rate, motion, shot cuts. Only `face_rate` has signal (raw ρ=0.115) |
| Metadata | 11d | log engagement signals + channel target encoding. Channel encoding dominates all signal |
| LLM scalars (GPT-4o-mini) | 8d | 8 memorability dimensions per video, saved in `llm_scalar_cache_v2.json` |

**LLM dimensions (v2):** `emotional_valence`, `human_presence`, `message_simplicity`, `novelty_surprise`, `narrative_arc`, `brand_prominence`, `repetition_hooks`, `direct_memorability`

---

## What We Tried — All Failed

| Approach | Result |
|---|---|
| CLIP ViT-L/14 visual embeddings (first 60 frames, mean/std/first/last pooling) | ρ ≈ 0.00 |
| CLIP zero-shot memorability prompts (cosine sim against prompt set) | ρ ≈ 0.00 |
| CLIP text embeddings (title + description) | ρ ≈ 0.00 |
| STT transcript semantic embeddings | ρ ≈ 0.00 |
| Frame stats — all except face_rate | ρ ≈ 0.00 |
| Channel-agnostic model (no channel encoding) | collapses to ρ ≈ 0.07 |
| Cross-target stacking | **Invalid for test** — zero channel overlap means brand predictions are near-constant, adding no signal |

---

## Best Results So Far

CV strategy: **Channel-stratified GroupKFold** — greedy bin-packing by channel size, Goldman Sachs (23% of data, 79 videos) gets its own dedicated fold, fully deterministic (no random seed).

| Target | Best feature combo | CV ρ |
|---|---|---|
| `memorability_score` | `meta + face_rate + LLM_v2` | **0.2729** |
| `brand_memorability` | `meta + face_rate` | **0.2199** |

---

## Core Unsolved Problem

All 9 test channels are unseen in training. Channel encoding falls back to the global mean for every test video — essentially a constant prediction. All content-based signals tried so far fail to discriminate within this narrow financial domain.

---

## Next Directions Not Yet Tried

1. **Heavier video-language models** — InternVideo2, VideoMAE, or similar models pretrained on richer video understanding tasks than CLIP. May capture memorability-relevant features CLIP misses in this domain.
2. **LLM direct score prediction** — Ask GPT-4o to directly predict a memorability score using few-shot examples from training set, rather than scoring intermediate dimensions.

---

## Scripts in Repo

| Script | Purpose |
|---|---|
| `extract_features.py` | Extracts visual (CLIP), text (CLIP), metadata, and frame stats features |
| `llm_score_cache.py` | Calls GPT-4o-mini to generate LLM scalar scores, saves to `llm_scalar_cache_v2.json` |
| `train_with_llm_score.py` | Current best training script — tunes Ridge/SVR/XGBoost across feature combos with channel-stratified CV |
| `eda.py` | Exploratory data analysis — score distributions, channel stats, metadata correlations |