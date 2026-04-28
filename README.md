# howtowin.lol

A League of Legends **world model for post-game teaching**.

The product is *Duolingo for gaming*: after each match, surface a small number of bite-sized lessons drawn from how Challenger-tier branches of the same situation actually played out. Mistake lessons ("from this state, your branch was usually losing") and strength lessons ("from this state, your branch was unusually winning") are co-equal — both are taught.

This is **not** a stats dashboard. It is a learned model of how LoL games evolve, used to retrieve teachable cohorts and explain the decision diff between what you did and what the cohort that won did.

## Authoritative documents

The canonical spec wins on any disagreement with older docs:

- `docs/superpowers/specs/2026-04-21-canonical-world-model-spec.md` — **canonical source of truth** for the architecture, locked items, allowed temporary proxies, and open questions.
- `docs/superpowers/plans/2026-04-22-canonical-world-model-implementation-overhaul.md` — implementation queue (5 phases) for moving from current proxies to the canonical target.
- `docs/superpowers/specs/2026-04-15-sequence-model-design.md` — original sequence-model design and milestones M1–M6 (historical, but milestones are still the ladder).
- `docs/superpowers/specs/2026-04-15-plan-b-rssm-v1-design.md` — Plan B (RSSM v1) loss schedule, KL/free-bits, latent shape, leak/cold-player methodology.
- `docs/superpowers/specs/2026-04-17-m4-retrieval-check-design.md` — M4 retrieval contract: key, whitening, baselines, gate.
- `docs/superpowers/plans/2026-04-23-thesislabs-experiment-documentation.md` — root experiment definition and child ablations.
- `docs/superpowers/plans/2026-04-24-skill-causal-team-map.md` — remote skill-causal execution plan (rank diagnostics → skill-aware retrieval → counterfactual interventions → causal filter).

Older specs and plans (Plan A, 2026-04-15 sequence-model-design, Plan B amendments) are historical context. When they disagree with the canonical spec, the canonical spec wins.

## Architecture (canonical target)

### Three input streams (mandatory)

1. **Static game context** — champions, side, queue/region/time bucket, **numeric patch-conditioning vector**, and a tokenized representation that can be cross-attended at every recurrent step (not just used to seed `h_0`).
2. **Player priors** — per-`puuid` learnable residual embedding plus crafted player-history features (rank/LP, mastery, recent-form). Injected via actor-slot gathers into the dynamic stream.
3. **Dynamic sequence (hybrid time, mandatory)**:
   - **Anchor frames** — periodic snapshots of participant/team/objective state (rich, not the thin 6-feature proxy).
   - **Event tokens** — typed, ordered LoL events between anchors with **decision/outcome separation**:
     - *Decisions* (agentive, teachable): `ITEM_PURCHASED`, `SKILL_LEVEL_UP`, `WARD_PLACED`, `RECALL`, `ENGAGE`, `DISENGAGE`.
     - *Outcomes* (consequences): `CHAMPION_KILL`, `BUILDING_KILL`, `ELITE_MONSTER_KILL`.

### RSSM core (Dreamer-lineage)

At each anchor step `t`:

- **Deterministic state** `h_t = GRU(h_{t-1}, [z_{t-1}, action_t])` — `h_t ∈ ℝ^512`. Recurs over **event tokens** between anchors (the canonical target; today's code still uses a pooled `ActionSummarizer` proxy).
- **Prior** `p(z_t | h_t)` — Gaussian, latent dim 32.
- **Posterior** `q(z_t | h_t, o_t)` — Gaussian, conditioned on the observation at `t`.
- **KL regularization** with **free bits** (0.5 nats per dim) and KL weight `0.01` to prevent posterior collapse without forcing it.

### Shared anchor representation

`r_t = [h_t || z_t]` is the **single shared anchor representation** consumed by all downstream heads and by retrieval. The retrieval key is `[h_t || μ_q(z_t)]` (544-dim total). **Heads and retrieval must read from the same representation** — otherwise training and retrieval are misaligned.

### Four decoder heads

All read from `r_t`:

1. **Next-event** (factorized): `event_type × actor × target` plus gated payload subheads. Forces the latent to encode who-does-what-to-whom, not just statistics.
2. **Next-frame features** — reconstruct the next anchor's frame to anchor state fidelity.
3. **Outcome** — final win/loss probability per team.
4. **Next-decision** — per-participant decision-token predictions (used as a teaching readout, **not** as a control input; see skill-causal plan).

Approximate Plan B loss weights (from `2026-04-15-plan-b-rssm-v1-design.md`): outcome 0.35, next-event 0.35, next-decision 0.15, next-frame 0.10, KL 0.05, plus a small (0.05/0.03/0.02) prior-rollout reconstruction term over 3 imagined steps.

### Imagination rollouts

Because the model is RSSM-style, we can roll the prior forward to imagine alternate futures from any anchor. This is what makes the **skill-causal counterfactual surface** possible: synthetic decision-token interventions on the rollout input path, scored by divergence in predicted outcome distributions.

## Downstream: retrieval (M4) and teaching (M5)

### M4 retrieval-check

The world model is only useful if its anchor states **cluster by outcome more than chance and at least competitively versus a hand-crafted frame-feature baseline**.

- **Key:** `[h_t || μ_q(z_t)]`, per-dim z-score whitened on the training corpus.
- **Backend:** PyTorch `cdist` + `topk` on GPU (corpus ≈ 1.16M anchor rows × 544 dim ≈ 2.5 GB, fits a 4 GB GTX 1650 SUPER).
- **Gate:** mean cohort outcome entropy at `k=64` ≥ **0.7 bits** on **both** holdouts (necessary, not sufficient — must also beat random and stay competitive vs `frame_features`).
- **Holdouts:**
  - **game-cold** — `SHA1(match_id) % 10 == 0`.
  - **player-cold** — every participant `puuid` falls outside the training player set (5-puuid-by-hash methodology).
- **Queries:** mid-game minutes (10–25), train-set self-retrieval excluded (M4-4tl).
- **Baselines:** `random-k`, `static_only` (model's static encoder alone), `frame_features` (≈90-dim hand-crafted per-anchor stats).

### M5 teaching surface

A teachable moment combines:

1. a **high-signal anchor** (decision-rich, branches diverge here),
2. a **polarity** (mistake or strength),
3. a **decision-diff explanation** ("the cohort that won bought X / warded Y / disengaged at Z; you didn't").

The skill-causal plan adds a **causal filter** (matching / DR / DML on `[h_t || μ_q(z_t)]` plus rank/role/patch covariates) so that surfaced candidates clear an overlap and agreement check before reaching the lesson surface.

## Current state (2026-04-27)

### What works

- Plan B (RSSM v1) trains end-to-end on a 12.8k-game corpus locally and a 51k-game corpus on the remote GPU host.
- Outcome AUC@15: game-cold ≈ 0.884, player-cold ≈ 0.841; leak probe ≈ 0.531 (clean).
- Frozen-minute-0 leak probe is clean.
- M4 retrieval pipeline, baselines, and reports are all wired.
- Skill-causal Step 5 (Gate E causal filter) closed; remote infra ready for Step 6 conditional branches.

### What is failing right now

- **M4 entropy gate FAIL** at headline `k=64` on the canonical-aligned checkpoint:
  - game-cold ≈ 0.692 → 0.698 bits (vs 0.7 threshold).
  - player-cold pending; expected ≈ 0.694.
  - Per-baselines comparison (`d6a` rerun): the model trails `frame_features` by ≈ 0.10 bits at `k=64` — **unchanged** from the pre-overhaul measurement. Higher-power eval (n_queries ≈ 139k vs 1480) tightened the estimate but didn't reveal a fresh regression. The model is *redistributing* entropy toward small `k` (better at `k=16,32`) and away from `k=64+`.

### Active temporary proxies (not the canonical target)

The current code still ships several proxies that the canonical spec calls out, each with an exit condition:

| Surface | Current proxy | Canonical target |
|---|---|---|
| Anchor schema | thin `10 × 6` frame bundle | rich participant/team/objective + spatial summaries |
| Static conditioning | seeds `h_0` only | tokenized + per-step cross-attention |
| Between-anchor dynamics | `ActionSummarizer` pooling | event-token recurrence |
| Head representation | mostly `z_t` only | all heads on `r_t = [h_t || z_t]` |
| Patch conditioning | `mode=scaffolding` zeros all but version triple | full numeric patch vector (`PATCH_VECTOR_DIM=1024`) |

The implementation overhaul plan (`2026-04-22-canonical-world-model-implementation-overhaul.md`) sequences these into 5 phases.

## Repo layout

```
code/model/
  plan_b_model.py        RSSM core + rollout (entry point for the v1 architecture)
  rssm.py                Deterministic h, prior/posterior z, KL + free bits
  encoders.py            Static/player/dynamic stream encoders
  heads.py               Outcome, next-event, next-frame, next-decision
  retrieval.py           [h_t || μ_q(z_t)] index + kNN query
  m4_eval.py             Cohort entropy at k_sweep, baselines comparison
  lesson.py              Lesson candidate generation per held-out match
  baselines/             static_only and frame_features retrieval baselines
  plan_b_train.py        Training loop, AMP, materialized-sample cache
  plan_b_eval.py         Outcome AUC + rollout top-k diagnostics
  player_features.py     Crafted player-history features
  tokenizer.py / tokens.py  Typed event payloads
  dataset.py             Anchor + event windowing, holdouts
  cli.py                 Subcommands: plan-b-train, plan-b-eval, retrieval-build, retrieval-eval, lesson, ...

data/
  retrieval/plan_b_index.pt   2.5 GB canonical-aligned anchor index
  splits/                     plan_a_holdout.txt, plan_b_cold_holdout.txt
  ...                         match SQLite stores (not in git, copied between hosts)

docs/
  superpowers/specs/    canonical spec + design records
  superpowers/plans/    implementation plans, ThesisLabs briefs, skill-causal map
  m4_retrieval_eval_report_*.md, plan_b_eval_report_*.md
```

Heavy artifacts (corpus DBs, checkpoints, retrieval index) are **not** in git. Weights live on Hugging Face (`hf` CLI, see `~/.claude/rules/huggingface.md`).

## Toolchain

- **Python:** `uv` + `pyproject.toml`. No `pip`. Run training with `uv run python -m model.cli ...`.
- **JS/TS:** `bun` for any future site/web work.
- **Web frontend (when one exists):** prefer Rust + WebAssembly (Leptos / Dioxus / Yew). See `~/.claude/rules/web-wasm.md`.
- **Task tracking:** `bd` (beads) — never TodoWrite or markdown TODO files.
- **GPU:** training runs on a remote WSL GPU host (1650 SUPER class). Hard runtime gate ≤ 24h on the 51k-game corpus, soft target ≤ 20h. Local 12.8k-game corpus is for development only; sizes there are not authoritative for production runtime planning.

## Common commands

Run from `code/`:

```bash
uv run python -m model.cli plan-b-train --epochs 30
uv run python -m model.cli plan-b-eval
uv run python -m model.cli retrieval-build --baselines
uv run python -m model.cli retrieval-eval --baselines
uv run python -m model.cli lesson --match-id NA1_5439588777 --team blue
```

## Milestones (status)

| ID | Description | Status |
|---|---|---|
| M1 | Plan A baseline classifier + data pipeline | done |
| M2 | Plan B RSSM v1 trains and produces outcome signal | done |
| M3 | Outcome AUC + rollout diagnostics + leak probe clean | done |
| M4 | Retrieval entropy gate ≥ 0.7 bits at k=64, both holdouts | **failing by ≈0.005 bits** |
| M5 | Lesson surface (anchor + polarity + decision-diff) | partial; gated on M4 |
| M6 | Productized teaching loop | not started |

The current bottleneck is M4 retrieval quality, which the canonical-overhaul implementation queue is designed to address. Until M4 passes, M5 lesson candidates are exposed but not validated.
