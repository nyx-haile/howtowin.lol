# Skill-Causal World Model — Remote Execution Team Map

**Date:** 2026-04-24  
**Status:** Execution brief for the remote compute server  
**Primary research input:** `.omx/specs/autoresearch-skill-causal-model/report.md`  
**Planning note:** this brief assumes the **full 51k-game corpus** lives on the remote compute server. Local corpus counts on this workstation are not authoritative for runtime or cohort-size planning.

## Purpose

Save the current skill-causal implementation plan as a repo document that can be pushed to the remote server and used as the execution brief there.

This plan incorporates the revised Section 9 direction:

- the model should **discover candidate decision families endogenously** through counterfactual sensitivity;
- causal estimators should act as a **validation filter** on model-surfaced candidates rather than an independent decision-family generator.

## Repo-grounded findings that shape the plan

1. **Rank is present, but indirect in the current world model.**
   - Rank/LP and player-history features exist in `code/model/player_features.py`.
   - Player features are encoded in `code/model/encoders.py` and injected into dynamic tokens through actor-slot gathers.
   - Anchor observations in `code/model/plan_b_model.py` do **not** directly include a skill/player-summary path.

2. **Retrieval is the main downstream bottleneck.**
   - `code/model/retrieval.py` index rows currently store only `match_id`, `anchor_minute`, and `blue_win`.
   - `code/model/lesson.py` queries retrieval without skill/rank filtering.
   - `docs/m4_retrieval_eval_report_2026-04-17.md` shows the model clears the entropy gate but still trails the frame-feature baseline.

3. **The revised Section 9 needs a new intervention surface.**
   - `NextDecisionHead` in `code/model/heads.py` is a decoder/readout, not a control input.
   - `rollout_prior_single` in `code/model/plan_b_model.py` advances using event-window embeddings, not decision logits.
   - Therefore the team must **not** implement "perturb NextDecisionHead logits" literally. The intervention mechanism must instead act on the rollout input path.

4. **Runtime infrastructure is already partially present and should be extended, not replaced.**
   - `code/model/plan_b_train.py` already includes runtime preflight, AMP, sparse secondary losses, and budgeted batching.
   - `code/model/dataset.py` and `code/model/sample_materialization.py` already contain materialized-sample cache plumbing.
   - `code/model/cli.py` does not yet expose sample-materialization controls for remote training workflows.

## Decision summary

### Principles

1. Measure rank usage before regularizing rank.
2. Fix retrieval confounding before changing the RSSM objective.
3. Model-surfaced decision candidates must be causally filterable before they can reach lessons.
4. The 24-hour training budget is a hard architecture constraint.
5. Prefer additive, reversible changes over global objective churn.

### Chosen sequence

1. **Runtime & budget**
2. **Diagnostics**
3. **Skill-aware retrieval correction**
4. **Model-endogenous candidate discovery via intervention surface**
5. **Causal validation filter**
6. **Conditional model/objective changes only if earlier phases justify them**

### Explicitly deferred

- GRL on the full stochastic latent from day one
- full IRM on the RSSM ELBO
- CITRIS / CausalVAE-style model replacement

## Agent model

**Each step is a one-shot stateless agent.** Agents do not persist between steps; all state passes through the filesystem via explicit handoff artifacts listed under each step. The orchestrator reads each step's output before launching the next.

Short review passes use non-persistent specialist agents:

- `architect` for intervention-surface review (after Step 4)
- `verifier` for final evidence checks (before any Step 6 branch)

## Lane reference

| Lane | Goal | Primary files | Handoff artifact |
|---|---|---|---|
| **Lane 1 — Runtime & corpus budget** | Keep 51k-game training under 24h on the remote server | `code/model/plan_b_train.py`, `code/model/cli.py`, `code/model/dataset.py`, `code/model/sample_materialization.py`, `scripts/verify_b41.py` | `artifacts/runtime_preflight.json` |
| **Lane 2 — Diagnostics** | Measure whether rank is under-used or over-leaking | new diagnostics module, `code/model/plan_b_model.py`, `code/model/retrieval.py` (read-only probes), tests | `artifacts/rank_diagnosis.json` |
| **Lane 3 — Rank metadata + retrieval** | Add skill-aware retrieval without collapsing cohorts | `code/model/retrieval.py`, `code/model/m4_eval.py`, `code/model/lesson.py`, rank-band utility | `artifacts/retrieval_eval.json` |
| **Lane 4 — Counterfactual intervention surface** | Make model-endogenous candidate discovery architecturally real | `code/model/plan_b_model.py`, new intervention helper/module, tests | `artifacts/intervention_candidates.json` |
| **Lane 5 — Causal validation filter** | Validate model-surfaced candidates against observed data | new causal-eval module, lesson-ranking glue, tests | `artifacts/causal_filter_report.json` |

All artifact paths are relative to the repo root. Create the `artifacts/` directory before Step 1 if it does not exist.

## Sequential execution plan

Steps run in order. Each step must complete and its gate must pass before the next step launches. Do not run steps in parallel.

---

### Step 1 — Runtime & corpus budget (Lane 1)

**Gate A** must pass before any subsequent step launches a full training run.

**Scope**

- expose sample-materialization settings in `plan-b-train`;
- add a remote-run profile for the full corpus;
- ensure runtime preflight reports are suitable for the compute server.

**Key tasks**

1. Add CLI/train-loop controls for materialized samples.
2. Add a preflight mode aimed at the 51k corpus and `<24h` budget.
3. Preserve Turing-safe behavior:
   - FP16 AMP + scaler on 1650 SUPER-class hardware
   - no assumption that `torch.compile` is available/useful.
4. Capture cache hit-rate and projected runtime in logs.

**Handoff artifact** — write `artifacts/runtime_preflight.json`:

```json
{
  "projected_early_stop_hours": <float>,
  "cache_hit_rate": <float>,
  "gate_a_pass": <bool>
}
```

**Gate A** — `gate_a_pass == true` and `projected_early_stop_hours <= 24`.

---

### Step 2 — Rank-use diagnostics (Lane 2)

Depends on: Step 1 (uses runtime infrastructure).  
**Lane 2 touches `retrieval.py` in read-only probe mode only. It must not write to `retrieval.py`.**

**Scope**

- produce evidence on whether rank is currently under-used or over-leaking.

**Key tasks**

1. Add shallow probes for rank band from:
   - `player_emb`
   - `h_t`
   - `post_mu`
   - `prior_mu`
   - retrieval key `[h_t || μ_q(z_t)]`
2. Add rank swap tests.
3. Add rank/LP-only ablation tests.
4. Add collapse/rate monitors:
   - per-dim KL
   - active units
   - MI/rate proxy summaries

**Handoff artifact** — write `artifacts/rank_diagnosis.json`:

```json
{
  "classification": "under_use" | "over_leak" | "mixed_or_unclear",
  "probe_auc_by_layer": { ... },
  "swap_delta": <float>,
  "gate_b_pass": <bool>
}
```

**Gate B** — `gate_b_pass == true` and `classification` is one of the three valid values.

---

### [Orchestrator checkpoint — after Step 2]

Read `artifacts/rank_diagnosis.json`. Record the classification. This value determines whether Step 6 opens and which branch it takes. Do not open any Step 6 branch before Gate B passes.

---

### Step 3 — Skill-aware retrieval (Lane 3)

Depends on: Step 2 complete (Lane 2 read-only probes are done; `retrieval.py` is now free for writes).

**Scope**

- add rank-aware retrieval while preserving practical cohort sizes.

**Key tasks**

1. Enrich retrieval rows with:
   - rank band
   - role
   - side
   - optional patch/champion-class metadata if easy
2. Implement two-stage retrieval:
   - same-band first
   - widen to adjacent bands if effective `k < 32`
   - unrestricted fallback with downweighting for rank-distant rows
3. Add dual reporting:
   - unrestricted entropy
   - skill-aware entropy
4. Surface effective cohort size and widening policy in lesson/retrieval outputs.

**Handoff artifact** — write `artifacts/retrieval_eval.json`:

```json
{
  "median_effective_k": <float>,
  "skill_aware_entropy_at_64": <float>,
  "unrestricted_entropy_at_64": <float>,
  "auc_at_15": <float>,
  "gate_c_pass": <bool>
}
```

**Gate C** — `gate_c_pass == true`:
- effective median `k >= 32` (target `k ≈ 64`)
- `auc_at_15` regression vs baseline `<= 0.02`
- `unrestricted_entropy_at_64` regression vs baseline `<= 0.03`

---

### [Architect review — after Step 3]

Dispatch an `architect` specialist to review the intervention-surface design before Step 4 begins. The architect reads the plan and the current state of `code/model/plan_b_model.py`. The architect's output is advisory; the orchestrator decides whether to proceed.

---

### Step 4 — Counterfactual intervention surface (Lane 4)

Depends on: Step 2 (diagnostic findings inform design), Step 3 (rank bands available).

**Scope**

- make revised Section 9 real through rollout interventions.

**Key tasks**

1. Add an intervention surface on the **rollout input path**, not the readout path.
2. Prefer **synthetic decision-token interventions** over direct logit perturbation.
3. Support interventions for the existing decision vocabulary:
   - `ITEM_PURCHASED`
   - `SKILL_LEVEL_UP`
   - `WARD_PLACED`
   - `RECALL`
   - `ENGAGE`
   - `DISENGAGE`
4. Add a divergence scorer comparing perturbed vs base rollout outcome distributions.

**Handoff artifact** — write `artifacts/intervention_candidates.json`:

```json
{
  "candidates": [
    {
      "decision_type": <str>,
      "anchor_minute": <int>,
      "divergence_score": <float>,
      "rank_band": <str>
    }
  ],
  "gate_d_pass": <bool>
}
```

**Gate D** — `gate_d_pass == true`: interventions change rollout outcome distributions and candidate rankings are stable enough to inspect.

---

### Step 5 — Causal validation filter (Lane 5)

Depends on: Step 3 (rank bands), Step 4 (`artifacts/intervention_candidates.json`).

**Scope**

- validate model-surfaced candidates instead of enumerating families by hand.

**Key tasks**

1. Read `artifacts/intervention_candidates.json` as the candidate input.
2. Validate candidates using matching / DR / DML.
3. Use covariates including:
   - `[h_t || μ_q(z_t)]`
   - raw frame/macro state
   - rank band
   - role / champion context
   - patch
4. Emit accept/reject with overlap diagnostics and confidence.

**Handoff artifact** — write `artifacts/causal_filter_report.json`:

```json
{
  "accepted": [ { "decision_type": <str>, "anchor_minute": <int>, "confidence": <float> } ],
  "rejected": [ { "decision_type": <str>, "anchor_minute": <int>, "reason": <str> } ],
  "gate_e_pass": <bool>
}
```

**Gate E** — a candidate may reach the lesson surface only if:

1. model divergence is high;
2. overlap is acceptable;
3. matching / DR / DML agree on direction or at least do not contradict the candidate;
4. no obvious rank-confounding failure remains.

---

### [Verifier review — after Step 5]

Dispatch a `verifier` specialist to check `artifacts/causal_filter_report.json` before any Step 6 branch opens. The verifier confirms gate evidence is coherent and no confound is unaddressed.

---

### Step 6 — Conditional model/objective branch

**Only open after:**

- Gate B: diagnostic classification is `under_use` or `over_leak` (not `mixed_or_unclear`)
- Gate D: interventions are real
- Gate E: at least one candidate accepted
- Verifier sign-off

**Branch selection** — read `artifacts/rank_diagnosis.json`:

| Classification | Branch |
|---|---|
| `under_use` | Open model-change lane for **direct skill-conditioning** |
| `over_leak` | Open model-change lane for **targeted split-latent / MI / MMD** |
| Head-level rank instability | Test **V-REx / invariance+IB on heads** |
| `mixed_or_unclear` | Stop; reassess with human. Do not open any branch. |

---

## Architectural constraints for revised Section 9

### Do not implement this literally

> "vary each decision type in `NextDecisionHead`'s output distribution"

This is not a faithful intervention in the current code because `NextDecisionHead` is only a readout head.

### Implement this instead

Use **synthetic decision-token interventions** that alter the rollout input path already used by `rollout_prior_single`.

This keeps the discovery mechanism aligned with what the model can actually consume.

## Retrieval / rank-band contract

Use coarse environments first:

- Iron–Silver
- Gold–Platinum
- Emerald–Diamond
- Master+

Only test finer bands if coarse bands show adequate overlap and useful signal.

## Remote runtime contract

### Hardware target

- NVIDIA GTX 1650 SUPER / TU116 class GPU
- 16 GB system RAM
- Ryzen 5-class CPU with 12 threads

### Training budget

- **Hard gate:** projected early-stop time `<= 24h`
- **Soft target:** `<= 20h`

### Remote-run policy

1. Warm materialized samples before the full run when practical.
2. Keep retrieval build/eval as **separate post-train jobs**.
3. Log:
   - projected epoch minutes
   - projected early-stop hours
   - token/anchor/event efficiency
   - cache hit/miss rate

## Verification and go / no-go gates

| Gate | Condition | Blocks |
|---|---|---|
| **A — runtime** | `projected_early_stop_hours <= 24` | All subsequent steps |
| **B — diagnostics** | rank failure mode classified | Step 6 branch selection |
| **C — retrieval** | effective `k >= 32`; AUC/entropy regressions within bounds | Step 5 integration |
| **D — intervention reality** | interventions change rollout distributions; rankings stable | Step 5 |
| **E — lesson-surface filter** | divergence high, overlap acceptable, estimators agree, no rank confound | Lesson surface |

## Success metrics

Primary near-term metrics:

- `player_cold_m4_entropy_at_64`
- `game_cold_m4_entropy_at_64`
- `player_cold_auc_at_15`
- `game_cold_auc_at_15`
- leak probe AUC@15

Operational success for this brief means:

- the remote agent can execute Steps 1–5 from this document;
- runtime remains inside the compute budget;
- no Step 6 branch opens before Gates B, D, and E are all satisfied.

## Stop conditions

Stop and reassess if any of the following occur:

- Gate A fails: preflight projects `> 24h` early-stop runtime;
- Gate C fails: skill-aware retrieval collapses cohort sizes below usable levels;
- Gate D fails: interventions do not materially affect rollout distributions;
- causal validation routinely fails because surfaced candidates have no usable overlap;
- Step 6 branch is requested but `classification == "mixed_or_unclear"`.

## Recommended handoff note for the remote server

When transferring this document to the compute server, treat it as the execution source of truth for:

1. step ordering and gates,
2. handoff artifact schemas and paths,
3. revised Section 9 intervention design,
4. runtime gating.

Do **not** use local workstation corpus sizes to overrule the runtime or cohort-size assumptions in this brief.
