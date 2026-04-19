"""M5-MVP lesson generation.

Given a target game and team perspective, encode the game's mid-game
anchors with the trained Plan B model, query the M4 retrieval index per
anchor, and surface the two highest-entropy cohorts as lesson candidates:

- mistake anchor: player lost, cohort entropy is high (many cohort games
  on the winning side — decision differentiated them)
- strength anchor: player won, cohort entropy is high (many cohort games
  on the losing side from the same latent state)

This is the MVP inference surface. It does NOT yet decode "what" separated
winners from losers — that comes from the decoder heads / decision
inspection in a later slice.
"""
import os
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from torch.utils.data import DataLoader

from model.dataset import (
    MatchDataset, build_puuid_index, collate_games, load_split,
)
from model.retrieval import (
    encode_game_keys, query_index, load_index,
    DEFAULT_INDEX_PATH, MID_GAME_MINUTES,
)
from model.m4_eval import binary_entropy


CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "model_checkpoints",
)
DEFAULT_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "plan_b_full_best.pt")

COHORT_PREVIEW = 8  # how many cohort match_ids to include in the output


@dataclass
class AnchorCandidate:
    minute: int
    cohort_entropy: float
    cohort_winrate: float  # fraction of cohort on blue-win side
    cohort_size: int
    cohort_match_ids: list[str]
    polarity: str          # "mistake" | "strength"


@dataclass
class LessonResult:
    match_id: str
    team: str
    player_won: bool
    mistake_anchor: Optional[AnchorCandidate]
    strength_anchor: Optional[AnchorCandidate]
    checkpoint_sha: str
    index_checkpoint_sha: str
    n_mid_anchors: int


def _load_model(ckpt_path: str, device: str):
    """Load PlanBModel checkpoint. Returns (model, max_puuids, state_sha)."""
    # Imported here so Plan A / CLI imports aren't pulled through when the
    # lesson module is loaded in lightweight contexts (tests, tools).
    from model.plan_b_model import PlanBModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    max_puuids = ckpt.get("max_puuids", 20000)
    model = PlanBModel(max_puuids=max_puuids).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # Best-effort checkpoint fingerprint: file modification time if nothing better.
    sha = ckpt.get("git_sha") or str(os.path.getmtime(ckpt_path))
    return model, max_puuids, sha


@torch.no_grad()
def _encode_mid_anchors(model, match_id: str, puuid_index: dict,
                        exclude_match_ids: set[str], device: str,
                        mid_minutes: set[int]):
    """Encode the target game; return (keys, minutes, blue_win) for mid-game anchors only."""
    ds = MatchDataset(
        [match_id], puuid_index=puuid_index,
        exclude_match_ids=exclude_match_ids, cache_size=1,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_games)
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        keys, minutes, blue_win = encode_game_keys(model, batch)
        mask = torch.tensor(
            [int(m.item()) in mid_minutes for m in minutes], dtype=torch.bool,
        )
        return keys[mask].cpu(), minutes[mask].cpu(), int(blue_win.item())
    raise RuntimeError(f"MatchDataset produced no sample for {match_id}")


def generate_lesson(
    match_id: str,
    *,
    team: str = "blue",
    k: int = 64,
    index_path: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    device: Optional[str] = None,
    mid_minutes=None,
) -> LessonResult:
    """Produce mistake + strength anchor candidates for `match_id`.

    team: "blue" (participants 1–5) or "red" (participants 6–10).
    k: cohort size per anchor.
    """
    if team not in ("blue", "red"):
        raise ValueError(f"team must be 'blue' or 'red', got {team!r}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path or DEFAULT_CHECKPOINT_PATH
    index_path = index_path or DEFAULT_INDEX_PATH
    mid_set = set(int(m) for m in (mid_minutes or MID_GAME_MINUTES))

    model, max_puuids, ckpt_sha = _load_model(ckpt_path, device)
    bundle = load_index(index_path)

    # puuid_index must match what training saw. Rebuild from the train split
    # with the checkpoint's max_puuids, then exclude both holdouts plus the
    # target match so player-feature aggregates stay leak-clean.
    train_ids = load_split("train")
    holdout = set(load_split("holdout"))
    cold = set(load_split("cold"))
    exclude = holdout | cold | {match_id}
    puuid_index = build_puuid_index(train_ids, max_puuids=max_puuids)

    keys, minutes, blue_win = _encode_mid_anchors(
        model, match_id, puuid_index, exclude, device, mid_set,
    )
    n_mid = int(keys.shape[0])
    player_won = bool(blue_win) == (team == "blue")

    if n_mid == 0:
        return LessonResult(
            match_id=match_id, team=team, player_won=player_won,
            mistake_anchor=None, strength_anchor=None,
            checkpoint_sha=ckpt_sha, index_checkpoint_sha=bundle.checkpoint_sha,
            n_mid_anchors=0,
        )

    cohort_idx, _ = query_index(bundle, keys, k=k, device=device)
    cohort_labels = bundle.row_blue_win[cohort_idx].float()  # (Q, k)
    cohort_winrate = cohort_labels.mean(dim=1)               # (Q,)
    entropy = binary_entropy(cohort_winrate)                 # (Q,)

    # Orient "win" from the target player's perspective: when team=='blue'
    # a cohort row labeled 1 is a win for this player. When team=='red' the
    # winning rows are the 0s.
    if team == "blue":
        player_cohort_winrate = cohort_winrate
    else:
        player_cohort_winrate = 1.0 - cohort_winrate

    def _candidate(polarity: str) -> Optional[AnchorCandidate]:
        """Pick the highest-entropy anchor matching the target polarity."""
        # Mistake = player lost the game → teach from anchors where cohort
        # entropy is high (other games won from the same state).
        # Strength = player won → show where the cohort was balanced and
        # this game's branch was the minority winning one.
        if polarity == "mistake" and player_won:
            return None
        if polarity == "strength" and not player_won:
            return None
        if entropy.numel() == 0:
            return None
        best = int(torch.argmax(entropy).item())
        row_ids = cohort_idx[best].tolist()
        match_ids_preview = [
            bundle.row_match_id[i] for i in row_ids[:COHORT_PREVIEW]
        ]
        return AnchorCandidate(
            minute=int(minutes[best].item()),
            cohort_entropy=float(entropy[best].item()),
            cohort_winrate=float(player_cohort_winrate[best].item()),
            cohort_size=k,
            cohort_match_ids=match_ids_preview,
            polarity=polarity,
        )

    return LessonResult(
        match_id=match_id, team=team, player_won=player_won,
        mistake_anchor=_candidate("mistake"),
        strength_anchor=_candidate("strength"),
        checkpoint_sha=ckpt_sha,
        index_checkpoint_sha=bundle.checkpoint_sha,
        n_mid_anchors=n_mid,
    )


def lesson_to_dict(res: LessonResult) -> dict:
    """JSON-safe view of a LessonResult (dataclasses + nested dataclass)."""
    d = asdict(res)
    return d
