"""kNN retrieval over Plan B latent state.

Builds an index of [h_t || mu_q(z_t)] per training anchor, fits per-dim
whitening, and exposes batched cdist+topk queries. Backed by PyTorch
tensors only — no FAISS dependency. See spec
docs/superpowers/specs/2026-04-17-m4-retrieval-check-design.md.
"""
import os
import subprocess
import time
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader

from model.dataset import MatchDataset, collate_games

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

_SIGMA_FLOOR = 1e-6


class Whitener:
    """Per-dim z-score normalization. Params fit once on the corpus."""

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        self.mu = mu
        self.sigma = sigma

    @classmethod
    def fit(cls, corpus: torch.Tensor) -> "Whitener":
        mu = corpus.mean(dim=0)
        sigma = corpus.std(dim=0, unbiased=False).clamp(min=_SIGMA_FLOOR)
        return cls(mu=mu, sigma=sigma)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sigma

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


def save_index(bundle: IndexBundle, path: str = DEFAULT_INDEX_PATH) -> None:
    if dirname := os.path.dirname(path):
        os.makedirs(dirname, exist_ok=True)
    torch.save(
        {
            "corpus_white": bundle.corpus_white,
            "whitener": bundle.whitener.state_dict(),
            "row_match_id": bundle.row_match_id,
            "row_anchor_minute": bundle.row_anchor_minute,
            "row_blue_win": bundle.row_blue_win,
            "checkpoint_sha": bundle.checkpoint_sha,
            "code_sha": bundle.code_sha,
            "built_at": bundle.built_at,
        },
        path,
    )


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
        except Exception as e:
            # Skip pathological games rather than aborting the whole build.
            print(f"[build_index] skipping match {eligible[i]}: {e}")
            continue
        T = keys.shape[0]
        rows_list.append(keys.cpu())
        minutes_list.append(minutes.cpu())
        blue_win_list.extend([int(blue_win.item())] * T)
        match_id_list.extend([eligible[i]] * T)
        if (i + 1) % log_every == 0:
            print(f"[build_index] {i + 1}/{len(eligible)} games encoded")

    corpus_raw = torch.cat(rows_list, dim=0) if rows_list else torch.zeros(0, KEY_DIM)
    minutes_all = torch.cat(minutes_list, dim=0) if minutes_list else torch.zeros(0, dtype=torch.int64)
    blue_win_all = torch.tensor(blue_win_list, dtype=torch.int8)

    whitener = Whitener.fit(corpus_raw) if corpus_raw.shape[0] > 0 \
               else Whitener(mu=torch.zeros(KEY_DIM), sigma=torch.ones(KEY_DIM))
    corpus_white = whitener.apply(corpus_raw)

    return IndexBundle(
        corpus_white=corpus_white,
        whitener=whitener,
        row_match_id=match_id_list,
        row_anchor_minute=minutes_all,
        row_blue_win=blue_win_all,
        checkpoint_sha=checkpoint_sha,
        code_sha=_git_head_sha(),
        built_at=int(time.time()),
    )


@torch.no_grad()
def query_index(
    bundle: IndexBundle,
    queries_raw: torch.Tensor,
    *,
    k: int,
    device: str = "cpu",
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cohort_idx, dists) of shape (Q, k) each.

    queries_raw: (Q, KEY_DIM) — raw keys, whitened internally with the
        bundle's fitted Whitener.
    """
    Q = queries_raw.shape[0]
    if Q == 0:
        return (
            torch.zeros(0, k, dtype=torch.long),
            torch.zeros(0, k, dtype=torch.float32),
        )

    corpus = bundle.corpus_white.to(device)
    mu = bundle.whitener.mu.to(device)
    sigma = bundle.whitener.sigma.to(device)

    out_idx = torch.empty(Q, k, dtype=torch.long)
    out_d = torch.empty(Q, k, dtype=torch.float32)

    for start in range(0, Q, batch_size):
        end = min(start + batch_size, Q)
        qb = queries_raw[start:end].to(device)
        qb_white = (qb - mu) / sigma
        d = torch.cdist(qb_white, corpus)        # (b, N)
        d_top, idx_top = d.topk(k, dim=1, largest=False)
        out_idx[start:end] = idx_top.cpu()
        out_d[start:end] = d_top.cpu()

    return out_idx, out_d
