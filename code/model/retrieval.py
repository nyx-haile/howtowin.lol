"""kNN retrieval over Plan B latent state.

Builds an index of [h_t || mu_q(z_t)] per training anchor, fits per-dim
whitening, and exposes batched cdist+topk queries. Backed by PyTorch
tensors only — no FAISS dependency. See spec
docs/superpowers/specs/2026-04-17-m4-retrieval-check-design.md.

Step 3 / Gate C additions:
- Per-row rank-band metadata attached during :func:`build_index`.
- :func:`query_index_skill_aware` implements same-band-first retrieval with
  adjacent-band widening and out-of-band-penalty fallback. See
  ``docs/superpowers/plans/2026-04-24-skill-causal-team-map.md`` §Step 3.
"""
import os
import subprocess
import time
from dataclasses import dataclass, field
import torch
from torch.utils.data import DataLoader

from model.dataset import MatchDataset, collate_games
from model.rank_band import N_BANDS, game_band

D_H = 512
D_Z = 32
KEY_DIM = D_H + D_Z  # 544

INDEX_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "retrieval"
)
DEFAULT_INDEX_PATH = os.path.join(INDEX_DIR, "plan_b_index.pt")

MID_GAME_MINUTES = tuple(range(10, 26))  # 10..25 inclusive
K_SWEEP = (16, 32, 64, 128, 256)
HEADLINE_K = 64
HEADLINE_GATE_BITS = 0.7

SKILL_MIN_EFFECTIVE_K = 32
SKILL_OUT_OF_BAND_PENALTY = 1.25
SCHEMA_VERSION = 2

_SIGMA_FLOOR = 1e-6
_WHITEN_TILE = 100_000  # rows per float64 tile during fit/apply


class Whitener:
    """Per-dim z-score normalization. Params fit once on the corpus."""

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        self.mu = mu.to(dtype=torch.float64)
        self.sigma = sigma.to(dtype=torch.float64)

    @classmethod
    def fit(cls, corpus: torch.Tensor) -> "Whitener":
        # Fit in float64 (float32 whitening on the small, high-variance fixture
        # corpus was numerically noisy enough to make the zero-mean property
        # intermittently fail in tests). Tile the corpus to avoid materializing
        # a full-corpus float64 copy — at ~1M rows × 544 dims that transient
        # is ~5 GB and was the dominant build-time RSS spike.
        n, d = corpus.shape
        if n == 0:
            zeros = torch.zeros(d, dtype=torch.float64)
            return cls(mu=zeros, sigma=torch.ones(d, dtype=torch.float64))

        # Pass 1: chunked sum → mu (numerically equivalent to .mean()).
        sum_acc = torch.zeros(d, dtype=torch.float64)
        for start in range(0, n, _WHITEN_TILE):
            tile = corpus[start:start + _WHITEN_TILE].to(dtype=torch.float64)
            sum_acc.add_(tile.sum(dim=0))
        mu = sum_acc / float(n)

        # Pass 2: chunked sum of squared deviations → variance (two-pass form
        # for numerical stability over a one-pass sum/sumsq).
        sq_acc = torch.zeros(d, dtype=torch.float64)
        for start in range(0, n, _WHITEN_TILE):
            tile = corpus[start:start + _WHITEN_TILE].to(dtype=torch.float64)
            tile.sub_(mu)
            tile.mul_(tile)
            sq_acc.add_(tile.sum(dim=0))
        sigma = (sq_acc / float(n)).sqrt().clamp(min=_SIGMA_FLOOR)
        return cls(mu=mu, sigma=sigma)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype if x.is_floating_point() else torch.float32
        mu = self.mu.to(device=x.device)
        sigma = self.sigma.to(device=x.device)
        n = x.shape[0]
        if n == 0:
            return x.to(dtype=out_dtype)
        # Tile the float64 transform to avoid materializing a full-corpus
        # float64 copy of x. Output is preallocated in the target dtype on the
        # input's device.
        out = torch.empty_like(x, dtype=out_dtype)
        for start in range(0, n, _WHITEN_TILE):
            stop = start + _WHITEN_TILE
            tile = x[start:stop].to(dtype=torch.float64)
            tile.sub_(mu)
            tile.div_(sigma)
            out[start:stop] = tile.to(dtype=out_dtype)
        return out

    def state_dict(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "Whitener":
        return cls(mu=sd["mu"], sigma=sd["sigma"])


@torch.no_grad()
def encode_game_keys(model, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run model.forward on a single-game batch (B=1) and return:

    keys:       (T, KEY_DIM) — concat of [h_t, post_mu_t] per anchor.
    minutes:    (T,) int64 — anchor minute = round(token_timestamp / 60000).
    blue_win:   () int8 — game-level outcome (1 if blue won, else 0).
    """
    assert batch["tokens"].size(0) == 1, "encode_game_keys expects B=1"
    was_training = model.training
    model.eval()
    out = model(batch)
    model.train(was_training)

    T = out["n_anchors"]
    h = out["h"][0]                # (T, D_H)
    post_mu = out["post_mu"][0]    # (T, D_Z)
    keys = torch.cat([h, post_mu], dim=-1)  # (T, KEY_DIM)

    anchor_pos = batch["anchor_positions"][0]   # (T,)
    ts = batch["token_timestamps"][0]           # (L,)
    anchor_ts = ts.gather(0, anchor_pos.long()) # (T,) ms
    minutes = (anchor_ts / 60000.0).round().to(torch.int64)

    blue_win = batch["outcome"][0].to(dtype=torch.int8)
    return keys, minutes, blue_win


@torch.no_grad()
def encode_game_rank_band(batch) -> int:
    """Game-level rank band for a B=1 batch. Returns -1 if lobby is unranked.

    Uses the mode of ranked players' coarse bands (see ``model.rank_band``).
    Cheap — a read of ``batch["players"][..., 0]`` plus a histogram.
    """
    assert batch["players"].size(0) == 1, "encode_game_rank_band expects B=1"
    return int(game_band(batch["players"])[0].item())


@dataclass
class IndexBundle:
    corpus_white: torch.Tensor             # (N, KEY_DIM) float32
    whitener: Whitener
    row_match_id: list[str]                # per-row source match_id
    row_anchor_minute: torch.Tensor        # (N,) int64
    row_blue_win: torch.Tensor             # (N,) int8 in {0, 1}
    checkpoint_sha: str
    code_sha: str
    built_at: int                          # unix ts
    # Step 3 (Gate C) additions — default to None for backward-compat with
    # pre-schema-v2 bundles saved before skill-aware retrieval.
    row_rank_band: torch.Tensor | None = None   # (N,) int8 in {-1..N_BANDS-1}
    row_patch: torch.Tensor | None = None       # (N,) int32 (major*100+minor); -1 unknown
    schema_version: int = 1

    def has_rank_metadata(self) -> bool:
        return (
            self.row_rank_band is not None
            and self.row_rank_band.numel() == len(self.row_match_id)
        )


def save_index(bundle: IndexBundle, path: str = DEFAULT_INDEX_PATH) -> None:
    if dirname := os.path.dirname(path):
        os.makedirs(dirname, exist_ok=True)
    payload: dict = {
        "corpus_white": bundle.corpus_white,
        "whitener": bundle.whitener.state_dict(),
        "row_match_id": bundle.row_match_id,
        "row_anchor_minute": bundle.row_anchor_minute,
        "row_blue_win": bundle.row_blue_win,
        "checkpoint_sha": bundle.checkpoint_sha,
        "code_sha": bundle.code_sha,
        "built_at": bundle.built_at,
        "schema_version": bundle.schema_version,
    }
    if bundle.row_rank_band is not None:
        payload["row_rank_band"] = bundle.row_rank_band
    if bundle.row_patch is not None:
        payload["row_patch"] = bundle.row_patch
    torch.save(payload, path)


def load_index(path: str = DEFAULT_INDEX_PATH) -> IndexBundle:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return IndexBundle(
        corpus_white=raw["corpus_white"],
        whitener=Whitener.from_state_dict(raw["whitener"]),
        row_match_id=list(raw["row_match_id"]),
        row_anchor_minute=raw["row_anchor_minute"],
        row_blue_win=raw["row_blue_win"],
        checkpoint_sha=raw["checkpoint_sha"],
        code_sha=raw["code_sha"],
        built_at=raw["built_at"],
        row_rank_band=raw.get("row_rank_band"),
        row_patch=raw.get("row_patch"),
        schema_version=int(raw.get("schema_version", 1)),
    )


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def build_index(
    *,
    model,
    train_match_ids: list[str],
    exclude_match_ids: set[str],
    puuid_index: dict,
    device: str = "cpu",
    checkpoint_sha: str = "unknown",
    log_every: int = 100,
) -> IndexBundle:
    """Encode every training game's anchors, fit whitening, return bundle.

    train_match_ids: candidate corpus games.
    exclude_match_ids: subtracted from the corpus AND passed to MatchDataset
        for player-feature-leak discipline (matches Plan B training contract).
    """
    model.eval()
    original_device = next(model.parameters()).device
    model.to(device)
    try:
        return _build_index_inner(
            model=model,
            train_match_ids=train_match_ids,
            exclude_match_ids=exclude_match_ids,
            puuid_index=puuid_index,
            device=device,
            checkpoint_sha=checkpoint_sha,
            log_every=log_every,
        )
    finally:
        model.to(original_device)


def _build_index_inner(
    *, model, train_match_ids, exclude_match_ids, puuid_index,
    device, checkpoint_sha, log_every,
) -> IndexBundle:
    eligible = [m for m in train_match_ids if m not in exclude_match_ids]

    rows_list: list[torch.Tensor] = []
    minutes_list: list[torch.Tensor] = []
    blue_win_list: list[int] = []
    match_id_list: list[str] = []
    rank_band_list: list[int] = []

    if not eligible:
        empty = torch.zeros(0, KEY_DIM)
        zero_w = Whitener(mu=torch.zeros(KEY_DIM), sigma=torch.ones(KEY_DIM))
        return IndexBundle(
            corpus_white=empty,
            whitener=zero_w,
            row_match_id=[],
            row_anchor_minute=torch.zeros(0, dtype=torch.int64),
            row_blue_win=torch.zeros(0, dtype=torch.int8),
            checkpoint_sha=checkpoint_sha,
            code_sha=_git_head_sha(),
            built_at=int(time.time()),
            row_rank_band=torch.zeros(0, dtype=torch.int8),
            schema_version=SCHEMA_VERSION,
        )

    ds = MatchDataset(
        eligible, puuid_index=puuid_index,
        exclude_match_ids=exclude_match_ids,
        cache_size=1,  # we only touch each game once
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_games)

    for i, batch in enumerate(loader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        try:
            keys, minutes, blue_win = encode_game_keys(model, batch)
            band = encode_game_rank_band(batch)
        except Exception as e:
            # Skip pathological games rather than aborting the whole build.
            print(f"[build_index] skipping match {eligible[i]}: {e}")
            continue
        T = keys.shape[0]
        rows_list.append(keys.cpu())
        minutes_list.append(minutes.cpu())
        blue_win_list.extend([int(blue_win.item())] * T)
        match_id_list.extend([eligible[i]] * T)
        rank_band_list.extend([band] * T)
        if (i + 1) % log_every == 0:
            print(f"[build_index] {i + 1}/{len(eligible)} games encoded", flush=True)

    corpus_raw = torch.cat(rows_list, dim=0) if rows_list else torch.zeros(0, KEY_DIM)
    rows_list.clear()  # release ~2.5 GB of per-game tensors before whitening
    minutes_all = torch.cat(minutes_list, dim=0) if minutes_list else torch.zeros(0, dtype=torch.int64)
    minutes_list.clear()
    blue_win_all = torch.tensor(blue_win_list, dtype=torch.int8)
    rank_band_all = torch.tensor(rank_band_list, dtype=torch.int8)

    whitener = Whitener.fit(corpus_raw) if corpus_raw.shape[0] > 0 \
               else Whitener(mu=torch.zeros(KEY_DIM), sigma=torch.ones(KEY_DIM))
    corpus_white = whitener.apply(corpus_raw)
    del corpus_raw  # whitened copy is the keep; raw is no longer needed

    return IndexBundle(
        corpus_white=corpus_white,
        whitener=whitener,
        row_match_id=match_id_list,
        row_anchor_minute=minutes_all,
        row_blue_win=blue_win_all,
        checkpoint_sha=checkpoint_sha,
        code_sha=_git_head_sha(),
        built_at=int(time.time()),
        row_rank_band=rank_band_all,
        schema_version=SCHEMA_VERSION,
    )


@torch.no_grad()
def query_index(
    bundle: IndexBundle,
    queries_raw: torch.Tensor,
    *,
    k: int,
    exclude_match_ids: set[str] | None = None,
    device: str = "cpu",
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cohort_idx, dists) of shape (Q, k) each.

    queries_raw: (Q, KEY_DIM) — raw keys, whitened internally with the
        bundle's fitted Whitener.
    exclude_match_ids: corpus rows whose row_match_id is in this set are
        masked out before selecting top-k. Fetch k + |exclude| + 5 rows
        internally so the final k remain after masking. O(k·|exclude|) per
        query — negligible.
    """
    Q = queries_raw.shape[0]
    if Q == 0:
        return (
            torch.zeros(0, k, dtype=torch.long),
            torch.zeros(0, k, dtype=torch.float32),
        )

    n_exclude = len(exclude_match_ids) if exclude_match_ids else 0
    fetch_k = min(k + n_exclude + 5, len(bundle.row_match_id)) if n_exclude else k

    corpus = bundle.corpus_white.to(device)

    out_idx = torch.empty(Q, k, dtype=torch.long)
    out_d = torch.empty(Q, k, dtype=torch.float32)

    for start in range(0, Q, batch_size):
        end = min(start + batch_size, Q)
        qb = queries_raw[start:end].to(device=device, dtype=corpus.dtype)
        qb_white = bundle.whitener.apply(qb)
        d = torch.cdist(qb_white, corpus)                       # (b, N)
        d_top, idx_top = d.topk(fetch_k, dim=1, largest=False)  # (b, fetch_k)

        if exclude_match_ids:
            idx_cpu = idx_top.cpu()
            d_cpu = d_top.cpu()
            for qi in range(end - start):
                kept_idx: list[int] = []
                kept_d: list[float] = []
                for ci in range(fetch_k):
                    row_i = int(idx_cpu[qi, ci].item())
                    if bundle.row_match_id[row_i] not in exclude_match_ids:
                        kept_idx.append(row_i)
                        kept_d.append(float(d_cpu[qi, ci].item()))
                        if len(kept_idx) == k:
                            break
                # Pad if corpus too small after exclusion.
                while len(kept_idx) < k:
                    kept_idx.append(kept_idx[-1] if kept_idx else 0)
                    kept_d.append(kept_d[-1] if kept_d else 0.0)
                out_idx[start + qi] = torch.tensor(kept_idx, dtype=torch.long)
                out_d[start + qi] = torch.tensor(kept_d, dtype=torch.float32)
        else:
            out_idx[start:end] = idx_top.cpu()
            out_d[start:end] = d_top.cpu()

    return out_idx, out_d


# ---------------------------------------------------------------------------
# Step 3 / Gate C — skill-aware retrieval
# ---------------------------------------------------------------------------


_STAGE_SAME_BAND = 0
_STAGE_ADJACENT = 1
_STAGE_UNRESTRICTED = 2


def _topk_smallest(d: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row-wise nearest distances/indices.

    Kept as a tiny helper so tests can assert which skill-aware stages are
    actually evaluated without monkeypatching ``torch.Tensor`` internals.
    """
    return d.topk(k, dim=1, largest=False)


def _effective_mask(row_band: torch.Tensor, q_band: int, max_dist: int) -> torch.Tensor:
    """True for rows whose band is within ``max_dist`` of ``q_band``.

    Unranked corpus rows (band == -1) are always included — they carry no
    signal to discriminate, so excluding them would needlessly shrink cohorts.
    An unranked *query* (q_band == -1) matches all rows.
    """
    if q_band < 0:
        return torch.ones_like(row_band, dtype=torch.bool)
    valid = row_band >= 0
    diff = (row_band - q_band).abs()
    return (~valid) | (diff <= max_dist)


@torch.no_grad()
def query_index_skill_aware(
    bundle: IndexBundle,
    queries_raw: torch.Tensor,
    *,
    k: int,
    query_rank_bands: torch.Tensor,
    exclude_match_ids: set[str] | None = None,
    device: str = "cpu",
    batch_size: int = 64,
    min_effective_k: int = SKILL_MIN_EFFECTIVE_K,
    out_of_band_penalty: float = SKILL_OUT_OF_BAND_PENALTY,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Two-stage skill-aware retrieval with adjacent widening and penalised fallback.

    queries_raw:         (Q, KEY_DIM) — raw (unwhitened) keys.
    query_rank_bands:    (Q,) long — band id per query in {-1..N_BANDS-1}.

    Returns ``(cohort_idx, cohort_d, info)`` where ``info`` has:
        effective_k_per_query     (Q,) int — rows eligible BEFORE widening
        widening_stage_per_query  (Q,) int — 0 same, 1 adjacent, 2 unrestricted
        in_band_fraction_per_query (Q,) float — fraction of final top-k that
                                  sit in the same band as the query.
    """
    if not bundle.has_rank_metadata():
        raise ValueError(
            "bundle has no row_rank_band metadata — rebuild the index with "
            "retrieval.build_index (schema v2+) before calling "
            "query_index_skill_aware."
        )

    Q = queries_raw.shape[0]
    if Q == 0:
        return (
            torch.zeros(0, k, dtype=torch.long),
            torch.zeros(0, k, dtype=torch.float32),
            {
                "effective_k_per_query": torch.zeros(0, dtype=torch.long),
                "widening_stage_per_query": torch.zeros(0, dtype=torch.long),
                "in_band_fraction_per_query": torch.zeros(0, dtype=torch.float32),
            },
        )

    assert query_rank_bands.shape == (Q,), (
        f"query_rank_bands shape {query_rank_bands.shape} != (Q,)={Q}"
    )

    corpus = bundle.corpus_white.to(device)
    N = corpus.shape[0]
    row_band = bundle.row_rank_band.to(torch.long).to(device)   # (N,)
    valid_row = row_band >= 0                                    # (N,)
    inv_valid = (~valid_row).unsqueeze(0)                        # (1, N)

    # Pre-compute exclude mask once over the entire corpus. Set membership in
    # Python is O(1) so this scan is fine; it's the per-query string lookups
    # we want to avoid in the hot loop.
    exclude_set = exclude_match_ids or set()
    if exclude_set:
        exclude_row = torch.tensor(
            [rid in exclude_set for rid in bundle.row_match_id],
            dtype=torch.bool,
        ).to(device)
    else:
        exclude_row = torch.zeros(N, dtype=torch.bool, device=device)
    not_excluded = (~exclude_row).unsqueeze(0)                   # (1, N)

    out_idx = torch.empty(Q, k, dtype=torch.long)
    out_d = torch.empty(Q, k, dtype=torch.float32)
    eff_k = torch.empty(Q, dtype=torch.long)
    stage = torch.empty(Q, dtype=torch.long)
    in_band = torch.empty(Q, dtype=torch.float32)

    penalty_t = torch.tensor(out_of_band_penalty, device=device, dtype=torch.float32)
    INF = float("inf")

    for start in range(0, Q, batch_size):
        end = min(start + batch_size, Q)
        b = end - start

        qb = queries_raw[start:end].to(device=device, dtype=corpus.dtype)
        qb_white = bundle.whitener.apply(qb)
        d_full = torch.cdist(qb_white, corpus)                   # (b, N)

        q_band = query_rank_bands[start:end].to(device=device, dtype=torch.long)  # (b,)
        unranked_q = (q_band < 0).unsqueeze(1)                   # (b, 1)

        # Band distance per (query, row). For unranked queries we'll override
        # to wildcard below — but the math still works because diff with
        # q_band=-1 may be large; we substitute via masks/where.
        diff = (row_band.unsqueeze(0) - q_band.unsqueeze(1)).abs()  # (b, N) long

        # Eligibility masks. Unranked queries match every row (wildcard).
        # Unranked corpus rows are always eligible (carry no rank signal).
        same_mask = unranked_q | inv_valid | (diff == 0)         # (b, N)
        adj_mask = unranked_q | inv_valid | (diff <= 1)          # (b, N)

        # After-exclude eligibility: combine with not_excluded.
        same_elig = same_mask & not_excluded                     # (b, N)
        adj_elig = adj_mask & not_excluded                       # (b, N)

        same_eligible_count = same_elig.sum(dim=1)               # (b,)
        same_kept = torch.clamp(same_eligible_count, max=k)      # (b,)
        ranked_q = ~unranked_q.squeeze(1)                        # (b,)
        same_stage_safe = (
            (~ranked_q | (same_kept >= min_effective_k))
            & (same_eligible_count >= k)
        )

        # Build per-stage distance tensors with +inf for ineligible rows.
        d_same = torch.where(same_elig, d_full, torch.full_like(d_full, INF))

        # Same-stage top-k is always needed. When the entire batch is provably
        # same-stage under the existing per-k rules and has enough same-stage
        # rows to avoid the pathological fallback, skip the adjacent and
        # unrestricted top-k work entirely.
        d_same_top, idx_same_top = _topk_smallest(d_same, k)

        # Effective cohort = rows eligible at the same-band stage BEFORE
        # exclude (matches semantics of the original implementation).
        eff_k_b = same_mask.sum(dim=1).to(torch.long)            # (b,)

        same_only_batch = bool(same_stage_safe.all().item())
        if same_only_batch:
            chosen_idx = idx_same_top
            chosen_d = d_same_top
            stage_b = torch.full((b,), _STAGE_SAME_BAND, dtype=torch.long, device=device)
        else:
            # Distance-penalised distances for the unrestricted fallback. For
            # ranked queries: penalty^|diff| where row is ranked, else 1. For
            # unranked queries: no penalty.
            penalty_pow = torch.pow(penalty_t, diff.to(d_full.dtype))  # (b, N)
            ones = torch.ones_like(penalty_pow)
            dist_penalty = torch.where(
                valid_row.unsqueeze(0).expand(b, N),
                penalty_pow,
                ones,
            )
            dist_penalty = torch.where(
                unranked_q.expand(b, N),
                ones,
                dist_penalty,
            )

            d_adj = torch.where(adj_elig, d_full, torch.full_like(d_full, INF))
            d_unr = torch.where(
                not_excluded.expand(b, N),
                d_full * dist_penalty,
                torch.full_like(d_full, INF),
            )

            # Batched top-k for fallback stages.
            d_adj_top, idx_adj_top = _topk_smallest(d_adj, k)
            d_unr_top, idx_unr_top = _topk_smallest(d_unr, k)

            adj_eligible_count = adj_elig.sum(dim=1)             # (b,)
            adj_kept = torch.clamp(adj_eligible_count, max=k)    # (b,)

            # Stage selection: escalate when the kept count would fall short
            # of min_effective_k, but only for ranked queries. This preserves
            # the current per-k semantics, including k < min_effective_k.
            escalate_to_adj = ranked_q & (same_kept < min_effective_k)
            escalate_to_unr = escalate_to_adj & (adj_kept < min_effective_k)
            stage_b = torch.where(
                escalate_to_unr, torch.full_like(escalate_to_unr, _STAGE_UNRESTRICTED, dtype=torch.long),
                torch.where(
                    escalate_to_adj, torch.full_like(escalate_to_adj, _STAGE_ADJACENT, dtype=torch.long),
                    torch.full_like(ranked_q, _STAGE_SAME_BAND, dtype=torch.long),
                ),
            )                                                     # (b,)

            # Gather chosen indices/distances by stage.
            stage_exp = stage_b.unsqueeze(1).expand(-1, k)        # (b, k)
            chosen_idx = torch.where(
                stage_exp == _STAGE_UNRESTRICTED, idx_unr_top,
                torch.where(stage_exp == _STAGE_ADJACENT, idx_adj_top, idx_same_top),
            )
            chosen_d = torch.where(
                stage_exp == _STAGE_UNRESTRICTED, d_unr_top,
                torch.where(stage_exp == _STAGE_ADJACENT, d_adj_top, d_same_top),
            )

            # Pathological-fallback: if any selected distance is +inf (e.g.
            # tiny corpus where even the unrestricted-after-exclude pool is
            # < k), fall back to unrestricted with no exclude/no penalty for
            # those queries.
            if torch.isinf(chosen_d).any():
                d_full_top, idx_full_top = _topk_smallest(d_full, min(k, N))
                if d_full_top.shape[1] < k:
                    pad = k - d_full_top.shape[1]
                    d_full_top = torch.cat([d_full_top, d_full_top[:, -1:].expand(-1, pad)], dim=1)
                    idx_full_top = torch.cat([idx_full_top, idx_full_top[:, -1:].expand(-1, pad)], dim=1)
                inf_row = torch.isinf(chosen_d).any(dim=1)        # (b,)
                chosen_idx = torch.where(inf_row.unsqueeze(1), idx_full_top, chosen_idx)
                chosen_d = torch.where(inf_row.unsqueeze(1), d_full_top, chosen_d)
                stage_b = torch.where(inf_row, torch.full_like(stage_b, _STAGE_UNRESTRICTED), stage_b)

        # In-band fraction (NaN for unranked queries).
        chosen_band = row_band[chosen_idx]                        # (b, k)
        match = (chosen_band == q_band.unsqueeze(1)).to(torch.float32)
        in_band_b = match.mean(dim=1)
        in_band_b = torch.where(
            unranked_q.squeeze(1),
            torch.full_like(in_band_b, float("nan")),
            in_band_b,
        )

        # Single CPU sync per batch — no per-query .item() in this path.
        out_idx[start:end] = chosen_idx.cpu()
        out_d[start:end] = chosen_d.cpu()
        eff_k[start:end] = eff_k_b.cpu()
        stage[start:end] = stage_b.cpu()
        in_band[start:end] = in_band_b.cpu()

    info = {
        "effective_k_per_query": eff_k,
        "widening_stage_per_query": stage,
        "in_band_fraction_per_query": in_band,
    }
    return out_idx, out_d, info
