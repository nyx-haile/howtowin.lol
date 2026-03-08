# howtowin.lol — Project Completion Plan

## Context

howtowin.lol is a LoL stat tracker that normalizes in-game stats across every axis of the game to give factual, evidence-based insights — not generic build advice, but specific, quantified statements like "your vision at 10 minutes is in the 20th percentile for your rank and has 0.7 correlation with losses." The ultimate goal is a match outcome prediction engine and per-player study system.

The pipeline is ~30% complete: Riot API client and Redis queuing work, data processing is blocked, frontend is a skeleton. We need to build the inference engine first (data model, stat computation, correlation analysis) then surface it in the UI.

**Long-term architecture assumption**: An AI research lab will train models on this corpus. Design the data pipeline to produce clean, structured tensors suitable for model training — every schema and feature decision should be made with "will this be a good training dataset?" in mind.

---

## The Inference Engine (Core of the Project)

This is the heart of howtowin.lol. Everything else — the website, the API, the UI — just surfaces what the engine computes.

### What We're Computing

**1. Per-Frame Feature Vectors**

At every 60-second frame for each of the 10 players, extract a fixed-length feature vector. Normalize across rank tier and patch to remove confounds. Target vector shape: `(num_games, T_max, 10, F)` where F is the number of features per player.

Raw per-player stats from `participantFrames`:
- `currentGold`, `totalGold`, `xp`, `level`
- `minionsKilled`, `jungleMinionsKilled` (combined: CS)
- `position.x`, `position.y`
- `timeEnemySpentControlled`
- Derived: `cs_per_min`, `gold_per_min`, `xp_per_min`

Derived team-relative stats (computed within the same game):
- `gold_diff_vs_lane_opponent` — gold minus the opponent in the same role
- `xp_diff_vs_lane_opponent`
- `cs_diff_vs_lane_opponent`
- `team_gold_total` / `enemy_gold_total` → `team_gold_lead`
- `team_level_sum` / `enemy_level_sum` → `team_level_lead`

Events contribute time-series features per player:
- `kills`, `deaths`, `assists` (cumulative at each frame)
- `vision_score` (estimated from WARD_PLACED and WARD_KILL events)
- `item_power_score` — estimated power level of current item set (using Dragon item data for gold cost, component relationships, and completed item multipliers)
- `objective_participation` — was this player in the fight for Dragon/Baron/Tower?

**2. Outcome Variable**

Binary: `win` (1 for winning team, 0 for losing team). Read from `GAME_END` event `winningTeam`. Stored alongside each match record.

**3. Stat Normalization**

For each stat at each minute bucket:
- Bucket by: rank tier (Iron/Bronze/Silver/Gold/Plat/Diamond/Master/GM/Challenger), role (TOP/JGL/MID/BOT/SUP), patch version
- Compute rolling mean and std across the corpus
- Normalized value: `(raw - mean) / std` → mean 0, std 1
- Store raw AND normalized vectors so the site can display both

This normalization makes cross-player and cross-game comparisons meaningful.

**4. Win Correlation Analysis**

For each normalized stat at each minute bucket, compute:
- Pearson correlation with `win` outcome across all games in the corpus
- Separate by rank tier and role
- Result: a "correlation matrix" `C[role][minute][stat]` — what matters, when, for whom

This is the core of the recommendation engine. Examples of insights it surfaces:
- "Jungle gold at 5 minutes has correlation 0.68 with win for Diamond+ — it's the single highest-impact early stat"
- "Bot ADC position.x entropy at 25 minutes correlates 0.42 with loss — meaning ADCs who overextend late lose more often"
- "Top lane CS diff at 15 minutes matters less for Gold tier (0.21 correlation) than for Master+ (0.44)"

**5. Win Probability Prediction**

At each game minute, given the current feature vector for all 10 players, predict the probability that team 1 wins. This is a binary classification problem on time-series data.

MVP approach (no ML lab required yet):
- **Logistic regression on normalized stat aggregates** at key minute breakpoints (5, 10, 15, 20, 25, 30): take the top N correlated stats, fit logistic regression, get `P(win | stats@t)`. Fast to compute, interpretable, works with moderate data volume.
- Produces: win probability curve for each game (how the game "swung" over time)

Future approach (AI research lab):
- **Sequence model (LSTM / Transformer)**: input is the full `(T, 10, F)` tensor for a game, output is `P(win)` at every timestep. This captures temporal dynamics — a team that's behind at 10 but has better item trajectories may actually be favored.
- **Graph-aware architecture**: represent the 5v5 as a graph (players as nodes, team membership and role matchups as edges) to model interactions. A GNN encoder + temporal attention head would be ideal.
- Training data target: 100k+ high-elo games with clean feature tensors. The pipeline below is designed to produce exactly this.

**6. Player Study System**

For a given player (PUUID), aggregate their stats across all processed games:
- Compute their per-role, per-minute normalized stat profile (their "fingerprint")
- Compare against the top-1000-player baseline at their rank
- For each stat where they are consistently in the bottom quartile, check its win correlation
- Surface: "Your top 3 below-average stats that most correlate with your losses" as the primary recommendation

Example output:
```
chaos#kotic — Diamond II ADC
Week of Feb 27, 2026

Identified weaknesses:
1. Vision score at 15m: 28th percentile for Diamond ADC (0.61 loss correlation)
2. Gold diff vs opponent at 10m: 31st percentile (0.54 loss correlation)
3. Team objective participation: 22nd percentile (0.48 loss correlation)

Your win probability curve average: Favored at 6-12 min, behind at 20+ min → you tend to win early games but lose long games
```

---

## Phase 1: Data Pipeline — Fix and Extend

**Goal**: Get clean feature tensors flowing from Riot API into queryable storage.

### 1a. Fix `code/parser.py`
- Remove `assert False` at line 100 (debug stop).
- Rename `mangle_event()` → `handle_event()` (replaces the current `pass` stub).
- Fix `handle_event()`'s `db` references: use `self.db` (a dict initialized per match in `handle_match()`).
- Remove terminal `assert False, "Unknown Event Type"` — replace with `self.log(f"Unknown event: {event['type']}")`.
- Replace ClickHouse INSERT in `handle_pframe()` with writes to SQLite (see 1b). ClickHouse was noted as insufficient for tensor matching (notes/24.8.12); SQLite is sufficient for the MVP and the corpus can be exported to Parquet for model training later.

### 1b. Data storage schema (`code/db.py` — new file)
Three SQLite tables in `../data/howtowin.db`:

```sql
CREATE TABLE games (
    match_id TEXT PRIMARY KEY,
    patch TEXT,
    queue_id INTEGER,
    game_duration_s INTEGER,
    winning_team INTEGER,  -- 100 or 200
    created_at INTEGER
);

CREATE TABLE frames (
    match_id TEXT,
    participant_slot INTEGER,  -- 1-10
    puuid TEXT,
    team_id INTEGER,           -- 100 or 200
    role TEXT,                 -- inferred from position
    timestamp_ms INTEGER,
    -- raw stats
    current_gold INTEGER,
    total_gold INTEGER,
    xp INTEGER,
    level INTEGER,
    cs INTEGER,
    pos_x INTEGER,
    pos_y INTEGER,
    -- derived / cumulative
    kills INTEGER,
    deaths INTEGER,
    assists INTEGER,
    vision_score INTEGER,
    item_ids TEXT,             -- JSON array of 7 item slots
    PRIMARY KEY (match_id, participant_slot, timestamp_ms)
);

CREATE TABLE players (
    puuid TEXT PRIMARY KEY,
    riot_id TEXT,
    rank_tier TEXT,
    rank_division TEXT,
    lp INTEGER,
    last_updated INTEGER
);
```

### 1c. Feature computation layer (`code/features.py` — new file)
After a match is parsed and stored, compute:
- Derived stats: `cs_per_min`, `gold_per_min`, `opponent_gold_diff`, `team_gold_lead`, etc.
- Correlation pre-computation: incremental update to a running stats table when new games arrive
- Store normalized values in a `frame_normalized` table (same schema as `frames` but with normalized floats)

### 1d. Fix `code/agents.py`
- Uncomment player scraper thread (line 28).
- After `handle_match()`, call `features.compute_match(match_id)` to compute derived stats.

### 1e. Expand `code/seed.py`
- Seed top 50 Challenger + Grandmaster players from NA, not just "chaos#kotic".
- Use `/lol/league/v4/challengerleagues/by-queue/{queue}` and grandmasterleagues endpoint.
- Add player rank/tier to the `players` table during seeding.

**Critical files**: `code/parser.py`, `code/db.py` (new), `code/features.py` (new), `code/agents.py`, `code/seed.py`

---

## Phase 2: Correlation & Prediction Engine (`code/engine.py` — new)

**Goal**: Compute the win correlation matrix and win probability model from stored data.

- `engine.compute_correlations()`: for each (role, minute_bucket, stat), compute Pearson correlation with `win` across all games. Store in a `correlations` SQLite table.
- `engine.train_win_predictor()`: fit a logistic regression model per minute bucket (using scikit-learn, already in the Python env likely, or installable). Serialize to a pickle in `../models/`.
- `engine.study_player(puuid)`: aggregate player's normalized stats, compare to corpus percentiles, return top weaknesses sorted by loss correlation.
- `engine.predict_match(match_id)`: using stored frames and trained model, produce per-minute win probability curve.

This module runs as a batch job — recompute after collecting N new games.

**Critical files**: `code/engine.py` (new)

---

## Phase 3: SvelteKit API Routes (Backend ↔ Frontend)

**Goal**: Expose engine output as HTTP endpoints.

Add `better-sqlite3` to `site/package.json`. Create:

- **`/api/player/[riotId]/+server.ts`**: look up PUUID from players table, call `engine.study_player()` equivalent in TS (or shell out to Python), return player study JSON.
- **`/api/games/[matchId]/[puuid]/+server.ts`**: return frames for the player from `frame_normalized` + win probability curve.
- **`/api/search/[query]/+server.ts`**: fuzzy search players by riot_id in the players table.

For the Python compute calls: use `child_process.execFile` to call a Python script that reads from SQLite and returns JSON to stdout. Fast enough for MVP; can be replaced with a persistent Python service later.

**Critical files**: `site/src/routes/api/` (new directory tree), `site/package.json`

---

## Phase 4: Frontend

### Layout fixes (`site/src/routes/+layout.svelte`, `site/src/lib/menu.svelte`)
- Fix sidebar: use CSS flexbox `width: 25%` instead of broken HTML `width="25%"` attribute.
- Fix menu: `padding: 100px` → `padding: 12px 16px`; remove hardcoded white background; add dark theme.
- Menu links: Home, **Explore**, About. Remove non-functional About/Contact until pages exist.

### Splash page (`site/src/routes/+page.svelte`)
- Dark, minimal. Project name + one-line description.
- Central search bar: `gameName#tagLine` → submits to `/explore?q=...`
- No data fetching needed; fully static.

### Explore page (`site/src/routes/explore/`)
- `+page.server.ts`: load player study data from API given `?q=riotId`.
- `+page.svelte`:
  - Search bar (pre-filled from URL param).
  - Player header: riot ID, rank, last updated.
  - **Weakness panel**: top 3 below-average stats with correlation badges and plain-English explanations.
  - Recent games list: champion, win/loss, duration, gold diff at 15, links to game view.

### Game detail view (`site/src/routes/games/[matchId]-[playerId]/`)
- `+page.server.ts`: load per-minute frame data + win probability curve.
- `+page.svelte`:
  - **Win probability curve** over game time (SVG line chart — no external lib needed).
  - Per-stat timeline: gold diff vs opponent, XP diff, CS diff.
  - Item purchase events rendered as icons on the timeline.
  - Outcome badge.

**Critical files**: `site/src/routes/+layout.svelte`, `site/src/lib/menu.svelte`, `site/src/routes/+page.svelte`, `site/src/routes/explore/` (new), `site/src/routes/games/[matchId]-[playerId]/+page.svelte`

---

## Deferred (Post-MVP)

- **Sequence model training** (LSTM/Transformer) at AI research lab — requires ~100k+ games first. Data pipeline is designed to produce export-ready Parquet tensors.
- **Graph neural network** for interaction modeling (5v5 player-player edges).
- **Riot account authentication** (OAuth flow with RSO) — for personalized dashboards.
- **Verified computation** — user-contributed compute for correlation batch jobs; defer until traffic warrants.
- **Live game prediction** — call Riot's Spectator API to show real-time win probability; requires low-latency model serving.
- **Champion/item recommendation** — requires normalized corpus + causal analysis (not just correlation).

---

## Critical Decisions

1. **SQLite for MVP, Parquet for training export**: Removes cloud dependency, fast for reads, easy to export to `(games, T, 10, F)` NumPy tensors for model training via `pandas.read_sql`.
2. **Logistic regression for win predictor**: Interpretable, fast, requires modest data volume (5k+ games). Replace with sequence model once lab capacity is available.
3. **No separate Python API server**: SvelteKit server routes shell out to Python for compute and read SQLite directly. One process, no networking between components.
4. **Feature vector design targets ML training**: every schema decision preserves raw tensors, timestamps, and normalization parameters so the corpus is immediately usable for model training.

---

## Verification

1. **Pipeline**: `python3 agents.py 2` — after processing 10 matches, verify rows in `data/howtowin.db` tables `games`, `frames`, `players`.
2. **Features**: `python3 -c "from features import compute_match; compute_match('NA1_5081314144')"` — verify `frame_normalized` rows written.
3. **Correlations**: `python3 -c "from engine import compute_correlations; compute_correlations()"` — verify `correlations` table populated.
4. **API**: `curl http://localhost:5173/api/search/chaos` — returns player JSON.
5. **Explore page**: `http://localhost:5173/explore?q=chaos%23kotic` — weakness panel and recent games render.
6. **Game view**: click a game — win probability curve and stat timelines render.

---

## Branch
Develop on: `claude/review-project-plan-p9CHp`
