# Kaggle Streaming Database ADR

**Status:** Proposed

**Date:** 2026-04-30

## Status

Proposed.

## Purpose

This ADR will decide the lowest-cost way to distribute and read this repo's game corpus for daily or weekly batch model runs over filtered shards. Kaggle is one candidate, not the default answer.

The decision must stay at the architecture and policy level. It must not create implementation work inside this ADR task.

## Decision

Ranked final recommendation:

1. **Primary live storage:** Use immutable compressed shards plus a small mutable manifest/index in Hugging Face Buckets or S3-compatible object storage such as Cloudflare R2 or Backblaze B2, optimized for lowest recurring cost. Pick the concrete provider during future execution by checking current storage price, bandwidth behavior, free-tier limits, private-bucket support, and batch-compute locality. The live corpus must not be a monolithic SQLite upload or a Kaggle dataset version stream.
2. **Snapshot/archive distribution:** Use Kaggle only for periodic public snapshots or releases if Riot/platform redistribution checks, privacy review, and publication-size checks pass. Hugging Face Datasets repos may also publish public streamable or versioned snapshots, especially when Parquet-style public access is useful. Neither Kaggle nor an HF Datasets repo should be the authoritative mutable live store for daily or weekly append/prune.
3. **Compute-host strategy:** Use stateless cheap batch compute that pins a manifest, reads selected shards, writes approved derived outputs, and exits. Hugging Face Jobs and Modal are the primary candidates when repeatability or reliability matters. Colab and Kaggle notebooks are lower-reliability experiment paths for manual or snapshot-bound runs. This ADR rejects always-on serving as the default because the workload is batch model work, not interactive low-latency API traffic.
4. **Update/prune model:** Append by adding immutable shard files and publishing a new manifest version. Prune by excluding shards from the manifest first, then compact or physically delete later only after pinned readers, rollback windows, snapshots, derived artifacts, provider retention, and privacy obligations allow it. Readers must pin manifest versions so stale readers are reproducible instead of mixed across live updates.

This is a decision, not a request to research more before choosing a direction. Future research is limited to provider-specific execution checks, pricing validation, permissions, and migration design before any implementation begins.

## Context / current state

The active normalized corpus is the SQLite database at `data/howtowin.db`. It is approximately 64 GB, is overrideable through `HOWL_DB_PATH`, and its default path and connection helpers are defined in `code/db.py`.

The raw Riot match and timeline corpus is the SQLite database at `data/raw_matches.db`. It is approximately 3.5 GB, contains 51,035 raw matches, is overrideable through `HOWL_RAW_DB_PATH`, and its default path and raw-store helpers are defined in `code/raw_db.py`.

Ingestion, corpus growth, and maintenance are file-local workflows today. Relevant paths are `code/fetch.py` and `code/parser.py` for Riot/queue ingestion and parsing, plus `code/grow_games.py`, `code/merge_db.py`, `code/reparse.py`, and `code/migrate_indices.py` for corpus growth and maintenance. The current research did not find a dedicated database prune or retention script. Any prune model in this ADR must be conceptual until verified before execution.

Model loading and artifacts assume local files. Relevant paths are `code/model/dataset.py`, `code/model/tokenizer.py`, `data/splits/`, `data/materialized_samples/plan_b`, `data/retrieval/plan_b_index.pt`, `data/retrieval/cache/`, and `data/model_checkpoints/`.

Deployment and sync also assume local files today. `scripts/sync-gpu.sh` rsyncs databases, splits, and checkpoints to the GPU host. The site storage evidence currently validated for this ADR is Cloudflare/D1 waitlist storage through `site/src/app.d.ts` and `site/src/routes/api/waitlist/+server.ts`. Earlier planning mentioned `site/src/lib/server/python.ts`, but current validation did not confirm it as active game-data storage; this ADR should rely on the Cloudflare/D1 waitlist paths unless the bridge is reverified.

The starting point is therefore a local SQLite corpus with local model artifacts, rsync-based GPU movement, and no confirmed database retention workflow. Any streaming recommendation must explain how it changes storage, reading, freshness, and pruning without changing source code, tests, uploads, deployments, schema, or proof-of-concept files in this ADR deliverable.

## Definitions of streaming levels

This ADR uses "streaming" as a spectrum, not a single capability. The storage choice must name which level it supports, because each level has different cost, freshness, and compute implications.

1. **Full dataset download:** Each run downloads or mounts the full corpus snapshot before training or evaluation. This is operationally simple, but it makes the current 64 GB normalized corpus the minimum movement unit and becomes costly or slow at 10x and 100x scale.
2. **File-level shard fetch:** The corpus is split into immutable shards, and a batch job fetches only the shard files selected by a manifest, split, patch, date window, queue, region, or other coarse filter. This matches the expected filtered shard scan workload.
3. **Range/object reads:** The reader can request byte ranges or individual objects inside a larger object store layout. This can reduce repeated transfers when only part of a shard is needed, but it depends on provider support, file format, and predictable access patterns.
4. **Predicate pushdown:** The storage format and query engine can skip row groups, columns, or partitions before data reaches the model job. This is a stronger form of filtering than file-level shard fetch and usually needs a columnar layout such as Parquet plus metadata that matches the training filters.
5. **Token/sample generator streaming:** The model job receives already filtered training samples, anchor windows, or token batches from a generator without first materializing the full corpus locally. This is the closest fit for long-term model training ergonomics, but it requires loader and manifest work outside this ADR.

## Decision drivers

Ranked drivers for this ADR:

1. **Lowest recurring cost:** The target budget is `$0/month`. A `<$10/month` option is an acceptable fallback only when it removes a clear operational blocker, such as mutable manifests or small object storage needs. Any `>$10/month` design requires explicit justification and should be treated as a fallback, not the default recommendation.
2. **Batch throughput for filtered shard scans:** The primary workload is daily or weekly model runs that scan selected shards. The design should minimize unnecessary full-corpus transfers and make selected-shard reads cheap enough to repeat.
3. **Daily or weekly freshness:** Daily updates mean smaller increments and fresher data, but more orchestration events and more manifest churn. Weekly updates mean larger loads and staler data, but lower orchestration frequency and fewer moving parts.
4. **Cold-start tolerance:** Batch jobs can tolerate cold starts when they preserve the lowest-cost path. The decision should not pay a standing serving bill only to reduce startup latency.
5. **Operational simplicity and rollback:** The corpus should be versioned so a run can pin a manifest or snapshot, retry safely, and roll back from bad data without mutating the only copy.
6. **Interactive serving latency:** Real-time API serving is not optimized in this ADR. It matters only if a candidate architecture accidentally makes future serving impossible; it is ranked below cost, batch throughput, freshness, and rollback.

Workload assumptions: runs are batch model runs, not live request handlers. Data freshness is daily or weekly. Most reads scan filtered shards. Cold starts are acceptable. The ADR does not optimize for real-time API serving.

## Non-goals

This ADR is not an implementation task. It forbids changes to source code, tests, database schema changes, upload tooling, deployment configuration, model artifacts, data files, and notebooks.

This ADR must not perform uploads to Kaggle, Hugging Face, R2, B2, S3, Modal, Colab, or any other platform.

This ADR must not create deployments or deployment automation.

This ADR must not create migration scripts, loader code, manifests, schemas, benchmark harnesses, or POCs.

This ADR must not mutate `code/`, `site/`, `scripts/`, `data/`, `DB/`, test files, database files, model checkpoints, credentials, or provider configuration.

This ADR may describe a conceptual migration path, but every execution step must remain future work and must be separately approved, verified, and scoped.

## Options considered

### Kaggle datasets and notebooks

Kaggle is not recommended as primary live storage.

Recommended Kaggle role: use Kaggle only as periodic public snapshot/archive distribution, and optionally for Kaggle notebook-native experimentation when reproducibility, discoverability, or community review matters. Kaggle should not be the mutable source of truth for the active corpus, append queue, prune policy, or live manifest.

As a snapshot/archive target, Kaggle can publish dated or release-tagged dataset versions of already prepared shard bundles. That role fits public, reproducible releases better than daily mutable live storage because readers can pin a dataset version and notebooks can run against the same released files. Public dataset downloads may be easier for consumers than private automation flows, but private datasets, automation, or account-scoped workflows still require Kaggle credentials, API tokens, or OAuth flows. That makes Kaggle acceptable for public archive distribution, but not a replacement for a low-friction mutable object store.

As primary live storage for the future append/prune workload, Kaggle is a poor fit for this ADR's filtered shard scan requirements:

1. Updates are versioned dataset operations rather than low-friction mutable object updates, so frequent append/prune would create dataset-version churn and release latency.
2. Consumers generally download whole datasets or selected files, not row-level or predicate-filtered streams. No official row-level, byte-range, predicate, or Parquet-style partial read support was confirmed in Kaggle docs available to this session; verify before execution.
3. Auth/token or OAuth requirements may matter for private data or automation. Public dataset downloads may be simpler, but private or automated publish/download paths still need credentials and should be verified before execution.
4. Kaggle API rate limits are dynamic, and fixed numeric limits are not published in the official docs available to this session. Numeric quotas must therefore be treated as verify-before-execution risks, not design guarantees.
5. Frequent append/prune through dataset versions risks churn, latency, and operational friction compared with publishing immutable shards plus a mutable manifest in object storage.

Kaggle notebooks remain useful as an optional experimentation surface for released snapshots, especially when the goal is notebook-native reproduction against a stable public dataset version. They should consume snapshots or selected archive files, not drive the authoritative append/prune lifecycle.

## Platform comparison matrix

This section compares low-cost storage and compute platforms without making the final Task 6 recommendation. Storage choice and compute-host choice are intentionally separate: the live corpus needs mutable, low-cost storage for immutable shards plus a mutable manifest, while batch model runs can execute on stateless compute that reads selected shards and exits.

Legend: **Strong** means a good fit for this ADR's lowest-cost filtered-shard batch workload; **Medium** means usable with caveats; **Weak** means not a primary fit; **Verify** means provider-specific limits or pricing must be checked before execution.

| Platform/provider | Storage mutability | Append/prune fit | Shard streaming fit | Auth complexity | Free/cheap compute fit | Cold-start/sleep risk | Persistence | Egress risk | 100x-scale risk |
|---|---|---|---|---|---|---|---|---|---|
| Hugging Face Buckets | Strong mutable object storage; docs describe Hub storage buckets for S3-style workloads. | Strong for adding immutable shards and replacing a small manifest; buckets are mutable and non-versioned, so manifest history still needs explicit versioning. | Strong for object-level selected-shard reads; range behavior and client details must be verified before execution. | Medium: HF auth and bucket permissions are simpler than multi-cloud IAM but still require tokens for private automation. | Weak as compute by itself; pair with HF Jobs, Spaces, Modal, Colab, or another runner. | Low for storage; compute cold start depends on paired runner. | Strong object persistence if bucket lifecycle is managed deliberately. | Verify: repeated public/private reads can still create bandwidth or quota exposure. | Medium: object storage scales better than dataset-version workflows, but metadata, listing, and egress must be monitored at 10x/100x. |
| Hugging Face Datasets repos | Medium: repo commits are versioned, but less mutable-object-store-like than buckets. | Medium for published shard snapshots and versioned manifests; weak for high-churn append/prune if every update is a repo commit. | Strong for public dataset streaming; HF docs describe streaming, HTTP/range requests, and Parquet partial reads. | Low to Medium: public reads can be simple; private repos and automation need tokens. | Weak as compute by itself; pair with external compute. | Low for storage. | Strong version history, but large repo/file guidance must be respected. | Verify: free public distribution is attractive, but heavy repeated training reads need bandwidth/quota review. | Medium to High: good for versioned releases and streamable Parquet, weaker as the authoritative mutable live store at 100x unless commit churn remains low. |
| S3-compatible object storage, including Cloudflare R2 and Backblaze B2 | Strong mutable object storage abstraction; many compute platforms can read S3-compatible endpoints. | Strong for immutable shard adds, manifest replacement, lifecycle rules, and later compaction; physical prune semantics depend on lifecycle configuration. | Strong for selected objects and usually range reads; exact behavior depends on provider and SDK. | Medium to High: access keys, bucket policies, endpoint configuration, and secret management add complexity. | Weak as compute by itself; strong as storage paired with HF Jobs, Modal, Colab, Fly.io, Render, or local runners. | Low for storage; compute cold start depends on paired runner. | Strong if lifecycle/versioning/backups are configured; verify provider durability model. | Medium: R2 can reduce common egress pain in some paths; B2 and generic S3 may charge egress depending on destination and volume, so verify before execution. | Medium: best general-purpose 100x live-storage candidate, but large scans can still make network and request costs dominant. |
| Kaggle datasets and Kaggle notebooks | Weak for mutable storage; datasets are better as versioned snapshots. | Weak for live append/prune because dataset versions add churn and latency. | Medium for selected-file snapshot reads; no official row-level, byte-range, predicate, or Parquet-style partial streaming support was confirmed in this session. | Medium: public use may be simple, but private or automated dataset workflows need Kaggle credentials or OAuth/API handling. | Medium for Kaggle notebooks experimentation against released snapshots. | Medium: notebooks are interactive/session-style and not a reliable always-on batch scheduler. | Medium for dataset versions as archive; weak for live mutable manifests. | Verify: API rate limits are dynamic and numeric limits are not treated as design guarantees. | High as primary live storage; acceptable as snapshot/archive plus notebook experimentation only. |
| Hugging Face Jobs | Weak as storage; job inputs should come from HF Buckets, HF Datasets, or S3-compatible storage. | Weak for storage lifecycle; useful for running append/prune jobs only if future implementation approves that. | Strong compute consumer for selected shards near HF-hosted data, subject to job configuration. | Medium: HF token and job configuration required. | Strong candidate for stateless batch runs when pricing and runtime fit. | Medium: jobs start on demand; cold starts acceptable for this ADR. | Weak local persistence; outputs should be written back to durable storage. | Medium: co-locating with HF storage may help, but cross-provider reads need verification. | Medium: suitable if job quotas, pricing, and runtime remain within cheap batch constraints. |
| Hugging Face Spaces | Medium for app storage only with configured persistent storage; not a data lake. | Weak to Medium: useful for demos or small state, not authoritative shard append/prune. | Medium as a reader of selected shards; not optimized as a scheduled batch engine. | Medium: HF auth plus Space secrets. | Medium for demos or lightweight jobs; weaker than Jobs for batch reliability. | High on free/sleeping Spaces; paid persistent storage changes cost profile. | Weak without persistent storage; Medium with paid persistent storage. | Medium: external shard reads can hit bandwidth limits or hidden costs. | High for primary corpus storage; Medium to High for compute if workload exceeds Space limits. |
| Modal | Weak as primary storage; Volumes are useful but have semantics and limits, while CloudBucketMount can attach S3/R2/GCS-style storage. | Medium: good compute-side access to bucket-backed shards; Modal Volumes are not the preferred corpus source of truth. | Strong for compute over large sequential shard reads through cloud bucket mounts, subject to mount/provider behavior. | Medium: Modal secrets plus storage credentials. | Strong candidate for per-second cheap stateless batch compute when free/cheap credits and runtime fit. | Medium: cold starts are expected but acceptable for batch. | Medium for Modal Volumes; strong persistence should remain in external object storage. | Medium: reads from external buckets can create provider egress/request costs. | Medium: compute can scale, but storage should stay in buckets and 100x network cost must be modeled. |
| Colab | Weak as storage; notebooks should mount or download selected shards from durable storage. | Weak for live append/prune; manual/session style is not authoritative lifecycle management. | Medium: can read selected shards, but session limits, idle behavior, and IO variability make it less reliable. | Medium: notebook auth, mounted storage, and secrets are user/session dependent. | Strong for occasional `$0/month` or cheap exploratory batch runs; weak for reliable unattended schedules. | High: sessions can disconnect, idle, or vary by tier; numeric quotas are sparse and verify-before-execution. | Weak local persistence; outputs must be saved externally. | Medium: repeated downloads from object storage may dominate free compute value. | High for reliable 100x recurring runs; acceptable as ad hoc notebook-style execution. |
| Replicate | Weak as storage; predictions should consume external files or URLs. | Weak for corpus append/prune. | Medium for model inference-style predictions over selected inputs, but less natural for full training/evaluation shard scans. | Medium: API tokens and externally hosted inputs. | Medium to Weak: useful for hosted model inference, less ideal for arbitrary batch corpus scans. | Medium: prediction startup and rate limits apply. | Weak: outputs and inputs should live elsewhere. | Medium: file input transfer and prediction limits can become bottlenecks. | High for 100x data-scanning workloads; better reserved for model inference endpoints if needed later. |
| Fly.io | Medium for app-local volumes, but volumes are local and non-replicated. | Weak to Medium: app-local append/prune risks single-region/local-volume coupling; better as compute reading buckets. | Medium: can run workers that fetch selected shards, but app networking and volume locality must be managed. | Medium to High: app deploy config, secrets, and bucket credentials. | Medium: cheap app/workers possible, but not as directly batch-oriented as Jobs/Modal/notebooks. | Medium: machines can start/stop; acceptable for batch if orchestrated. | Medium: volumes persist locally but are not a replicated data lake. | Medium: cross-provider reads and app bandwidth need verification. | High for primary corpus storage; Medium for compute if reliability and scheduling are acceptable. |
| Render | Medium for app disks/workflows, but not a data-lake storage layer. | Weak to Medium: one-off jobs/workflows can mutate external storage, but Render disks should not be authoritative corpus storage. | Medium: jobs can read selected shards from external object storage. | Medium: service/job configuration plus secrets. | Medium: free/cheap tiers and one-off jobs/workflows may fit small batch jobs; verify cost and runtime. | High on free services that sleep; background jobs/workflows need pricing verification. | Medium for service disks; stronger persistence should remain external. | Medium: external data reads and bandwidth need verification. | High for primary storage; Medium to High for compute at 100x depending on runtime, scheduling, and network costs. |

Storage conclusion: the lowest-cost primary live-storage candidates are **Hugging Face Buckets** and **S3-compatible object storage, explicitly including Cloudflare R2 and Backblaze B2**, because they separate durable mutable object storage from compute and can support immutable shards plus a mutable manifest. **Hugging Face Datasets repos** remain a strong candidate for public streamable snapshots or versioned Parquet-style releases, but are less clearly the best live mutable store if daily append/prune creates frequent commit churn. Kaggle remains snapshot/archive and notebook experimentation only.

## Cost model for current, 10x, and 100x scale

These are planning estimates, not exact forecasts. Compression, sharding, deduplication, schema changes, pruning policy, and future data mix may change actual size. The estimates are useful for comparing modes before provider-specific pricing is chosen.

| Mode | Normalized active SQLite estimate | Raw SQLite estimate | Planning implication |
|---|---:|---:|---|
| current | ~64 GB from `data/howtowin.db` | ~3.5 GB from `data/raw_matches.db` | Full download is possible but already large enough that repeated transfers can dominate free compute sessions. File-level shard fetch is preferred for repeated batch scans. |
| 10x | ~640 GB if size scales linearly | ~35 GB if size scales linearly | Full-corpus movement becomes a poor default. Storage layout should favor immutable compressed shards, manifests, and filtered reads. Weekly loads reduce orchestration but create larger transfer bursts. |
| 100x | ~6.4 TB if size scales linearly | ~350 GB if size scales linearly | The design must assume full downloads are exceptional. Egress, bandwidth caps, cold-start transfer time, and free-tier reliability become primary risks. Predicate pushdown or generated sample streaming becomes much more valuable. |

Cost assumptions for this ADR:

1. **Storage cost:** Prefer public or free storage for immutable snapshots and shards when redistribution rules allow it. If mutable object storage is needed for manifests, small indexes, or private staging, keep the design within the `<$10/month` acceptable fallback unless a later task justifies more.
2. **Egress and bandwidth risk:** Even when storage is free, repeated reads of 64 GB, 640 GB, or 6.4 TB can be the hidden cost. A candidate that forces full dataset downloads for every run is weaker than one that supports shard selection, range reads, predicate pushdown, or token/sample generator streaming.
3. **Compute runtime and cold starts:** Free or cheap compute is acceptable if jobs can start cold, fetch selected data, run batch work, and exit. Paying for always-on interactive serving conflicts with the lowest-cost priority unless a later task proves it is needed.
4. **Free-tier reliability risk:** Free services can have throttling, rate limits, eviction, quota shifts, or variable availability. Unknown numeric quotas should be recorded as verification risks, not treated as guarantees.
5. **Freshness tradeoff:** Daily updates are better for smaller transfer increments and fresher training data. Weekly updates are simpler to schedule and reduce orchestration frequency, but they create larger shard batches and slower correction of bad ingests.

Budget thresholds:

1. `$0/month` target for the reference path.
2. `<$10/month` acceptable fallback for narrowly scoped mutable storage or reliability gaps.
3. `>$10/month` requires explicit justification, a clear reason cheaper options fail, and a reevaluation trigger.

## Recommended architecture

Keep the live corpus in low-cost mutable object storage as **immutable compressed shard files plus a small mutable manifest/index**, consumed by stateless batch compute. The preferred storage class is Hugging Face Buckets or S3-compatible object storage such as Cloudflare R2 or Backblaze B2. The provider should be chosen later by current cost, private-access needs, egress behavior, and compute locality, not by brand preference.

In this model, shards are append-only data objects. The manifest is the small mutable control plane that names active shard files, retired shard files, dataset version metadata, checksums, logical partitions, and coarse indexes used by batch jobs to select only the files they need. Compute hosts such as Hugging Face Jobs, Modal, Colab, or notebook-style runners should pin a manifest, fetch selected shards, write derived outputs deliberately, and exit. The corpus store remains separate from compute; Kaggle remains a snapshot/archive or notebook experimentation surface, not the primary live mutable corpus store.

The recommended default read path is file-level shard fetch for daily or weekly batch scans. Range/object reads and predicate pushdown are desirable future optimizations, especially at 10x or 100x scale, but they do not change the storage decision. Token/sample generator streaming remains future loader work outside this ADR.

Conceptual storage categories:

1. **Compressed raw shards:** raw Riot matches and timelines currently held in `data/raw_matches.db` become compressed raw shard objects for replay, reparsing, or audit use.
2. **Analytics shards:** normalized `games`, `players`, `frames`, `events`, and materialized statistics currently held in `data/howtowin.db` become analytics shard objects optimized for filtered batch scans.
3. **Versioned metadata artifacts:** split and holdout metadata currently under `data/splits/` becomes versioned metadata referenced by a manifest version so training/evaluation runs are reproducible.
4. **Derived sample cache artifacts:** `data/materialized_samples/plan_b` becomes a derived cache that can be regenerated from analytics shards or versioned separately when regeneration cost is high.
5. **Derived retrieval artifacts:** `data/retrieval/plan_b_index.pt` and `data/retrieval/cache/` remain derived retrieval indexes/caches, not live source-of-truth data.
6. **Model artifacts:** `data/model_checkpoints/` remains separate model artifact storage, not corpus shards.

Default shard keys should match filtered batch scans rather than entity-specific slicing. Partition by time window first, such as `date` or `week`, then by patch, queue, region, and tier when those fields are available and materially reduce scan size. High-cardinality filters such as champion, lane matchup, item build, or exact player cohort should be served through shard metadata, per-shard statistics, or secondary indexes rather than tiny per-champion shards unless future evidence shows those filters dominate cost and produce healthy shard sizes.

## Update/prune/versioning model

Updates mutate the manifest/index, not existing shard contents. Shards are immutable; manifests are versioned; readers pin the manifest version they run against.

1. **Append:** Add new immutable compressed shard files for the daily or weekly ingest window, validate their metadata and checksums conceptually, then publish a new manifest version that includes those shard files. Existing shard objects are not rewritten for ordinary append operations.
2. **Manifest versioning:** Every publish creates a new manifest version with a stable identifier, creation time, parent pointer or previous-version reference, active shard list, excluded shard list, and enough metadata for filtered selection. The mutable pointer, such as `latest`, should move only after the new manifest version is fully published.
3. **Stale reader behavior:** A batch run pins one manifest version at startup and continues with that version even if a newer manifest is published mid-run. This avoids mixed-corpus reads, makes retries reproducible, and lets long-running jobs finish without coordination locks.
4. **Prune:** Prune is logical first. The next manifest version excludes obsolete, bad, private, or out-of-retention shards from the active set. Physical deletion is a later safety step, not immediate proof that the data is gone from every cache, snapshot, or derived artifact.
5. **Physical deletion and retention:** Physical removal should happen only after no pinned readers, rollback requirements, snapshot/archive requirements, provider lifecycle settings, and redistribution obligations need the excluded shard. Provider lifecycle rules or manual deletion policies are future implementation details, not part of this ADR task.
6. **Compaction:** Scheduled compaction rewrites many small, obsolete, or partially pruned shards into optimized shard sets, then publishes a new manifest version that points to the compacted shards and excludes the superseded shards. Compaction is also the point where physical storage can be reclaimed when it is safe.
7. **Rollback:** Rollback moves the mutable manifest pointer back to a prior known-good manifest version. Because shard files are immutable, rollback should not require rewriting data; it should only restore a previous active shard/index view unless the underlying objects were already physically deleted.
8. **Index updates:** Small mutable metadata/index files may be published with each manifest version to accelerate time, patch, queue, region, tier, champion, or matchup filtering. These indexes are selectors over immutable shards, not authoritative copies of the corpus.

This model supports daily or weekly freshness without making any compute host stateful and without making Kaggle the live append/prune system.

## Compute-host fit

Compute candidates are evaluated as execution options for batch model runs over selected shards. They are not the primary storage decision. The expected pattern is: a job pins a manifest, reads only selected shard files from HF Buckets, S3-compatible object storage, or an HF Datasets snapshot, writes derived outputs to durable storage if needed, and exits.

| Compute option | Best ADR role | Fit for batch model runs over selected shards | Storage coupling | Cold-start/sleep risk | Persistence model | Cost/scale caveats |
|---|---|---|---|---|---|---|
| Hugging Face Jobs | Primary compute candidate when data lives on HF or job pricing fits. | Strong for stateless scheduled or ad hoc jobs that read shard manifests and exit. | Reads naturally from HF Hub assets and can also use external storage if configured. | Medium; on-demand startup is acceptable for daily/weekly batch work. | Treat local disk as ephemeral; write results to HF storage or object storage. | Verify job pricing, hardware availability, max runtime, and private-data handling before execution. |
| Modal | Primary compute candidate when per-second execution and cloud bucket mounts fit. | Strong for batch scans, especially when mounting S3-compatible buckets such as Cloudflare R2 or Backblaze B2. | Best paired with external buckets through CloudBucketMount; Modal Volumes should not become the live corpus source of truth. | Medium; cold starts are acceptable for batch. | Durable corpus remains in object storage; Modal outputs should be committed or uploaded deliberately. | Verify free/cheap credits, GPU/CPU pricing, volume limits, and external bucket egress at 10x/100x. |
| Colab | Low-cost exploratory or manual notebook-style execution. | Medium for occasional runs over a small selected shard set; weak for unattended recurring jobs. | Reads from HF Datasets, HF Buckets, S3-compatible URLs, or mounted drives depending on future implementation. | High; idle disconnects and sparse quota details make reliability uncertain. | Notebook VM storage is ephemeral; outputs must be copied to durable storage. | Good `$0/month` experiment path, not a dependable 100x scheduler without manual tolerance. |
| Kaggle notebooks | Snapshot-bound experimentation and reproducibility against Kaggle dataset releases. | Medium for notebook-native experiments on released snapshots; weak for live daily/weekly corpus processing. | Best coupled to Kaggle snapshot/archive datasets, not mutable live storage. | Medium to High; notebook session model is not an authoritative batch platform. | Outputs must be saved as notebook outputs or external artifacts, not as the live manifest. | Keep consistent with Task 3: useful for archive experimentation only. |
| Hugging Face Spaces | Optional demo/lightweight runner, not the default batch host. | Weak to Medium for batch jobs; Spaces are app-oriented and can sleep. | Can read from HF or external storage through secrets. | High on free sleeping Spaces; paid resources change the cost decision. | Persistent storage is optional and should not replace the corpus store. | Consider only if a later product/demo need justifies app hosting. |
| Render | Optional workflow/one-off-job host if reliability is more important than pure `$0/month`. | Medium for scheduled workers reading selected shards from object storage. | External object storage should be authoritative; Render disks are app-local. | High for free services that sleep; one-off jobs/workflows need pricing verification. | Write outputs back to object storage; do not rely on ephemeral job disk. | Verify free-tier sleep, workflow/job pricing, max runtime, and bandwidth before execution. |
| Fly.io | Optional app/worker host for custom orchestration. | Medium for workers reading selected shards; less direct than HF Jobs or Modal. | External object storage should be authoritative; Fly volumes are local and non-replicated. | Medium; machines can start/stop. | Local volumes may persist but should not be the corpus source of truth. | Verify VM, volume, bandwidth, and region costs before choosing it for recurring runs. |
| Replicate | Inference-oriented fallback, not a corpus scanning default. | Weak to Medium; better for model predictions than arbitrary large shard scans. | Inputs should come from external storage URLs or files. | Medium; prediction startup and rate limits apply. | Outputs are prediction artifacts, not durable corpus state. | Likely expensive or awkward for repeated 64 GB to 6.4 TB scans; verify before any inference-specific use. |

The selected compute strategy is stateless batch execution. The best low-cost compute-host candidates are **Hugging Face Jobs** and **Modal** for repeatable runs, with **Colab** or **Kaggle notebooks** as lower-reliability notebook-style experimentation paths. **Hugging Face Spaces**, **Render**, and **Fly.io** may fit app/workflow hosting if reliability or future UI needs justify them, but they should read from external storage rather than become the live corpus store. **Replicate** is not a primary batch-scan host because its product shape is closer to hosted predictions than data-lake compute.

## Security/auth/privacy and redistribution considerations

Security, auth, privacy, and redistribution constraints for the final recommendation:

1. **Public vs private data:** Public snapshots are easier to share through Kaggle or Hugging Face Datasets, but live mutable storage may need private buckets until Riot/platform redistribution rights, user privacy expectations, and derived-artifact exposure are verified.
2. **Provider tokens:** Kaggle, Hugging Face, S3-compatible buckets, Modal, Render, Fly.io, and notebook hosts all introduce provider tokens or access keys for private automation. No credentials, API tokens, OAuth material, bucket secrets, or generated config files should be committed to this repo.
3. **Bucket permissions:** The live corpus bucket should follow least-privilege permissions: write/publish access limited to future ingestion or release automation, read access limited to approved batch runners, and separate public snapshot permissions where redistribution is allowed.
4. **Credential storage:** Future execution should use provider secret stores, environment variables, CI secrets, or local ignored config. The ADR must not create credential files or document real secrets.
5. **Riot/platform redistribution checks:** Before publishing any raw or normalized corpus publicly, verify Riot API terms, platform policies, account identifiers, player privacy expectations, and whether raw timelines, PUUIDs, derived stats, model checkpoints, or retrieval indexes can be redistributed.
6. **Derived artifact leakage:** Retrieval indexes, sample caches, and model checkpoints can encode sensitive or restricted data even when they are not raw matches. Treat them as separate artifacts with their own publication review rather than assuming they inherit raw-shard permissions.
7. **Deletion expectations:** Manifest prune is not a privacy deletion guarantee by itself. If a deletion request or policy requires physical removal, future implementation must account for archived snapshots, caches, compacted shards, derived samples, retrieval indexes, checkpoints, and provider retention behavior.

## Risks and mitigations

1. **Privacy and redistribution risk:** Raw matches, timelines, PUUIDs, derived samples, retrieval indexes, and checkpoints may contain data that should not be public. Mitigation: keep live storage private by default until Riot/platform terms, account identifiers, player privacy expectations, and derived-artifact leakage are reviewed.
2. **Platform ToS, rate limit, and quota risk:** Kaggle, Hugging Face, Modal, R2, B2, Colab, Render, Fly.io, and other platforms can change terms, rate limits, free-tier behavior, or unpublished quota enforcement. Mitigation: treat numeric limits as verify-before-execution facts, keep storage and compute decoupled, and prefer portable S3-style shard layouts.
3. **Egress surprise risk:** A free or cheap storage tier can become expensive if repeated batch jobs read 64 GB, 640 GB, or 6.4 TB across providers. Mitigation: fetch only selected shards, co-locate compute with storage when practical, track bandwidth before committing to 10x or 100x operation, and avoid full-corpus downloads as the default.
4. **Free-tier instability risk:** Colab, Kaggle notebooks, Spaces, and other free compute paths can sleep, disconnect, throttle, or change availability. Mitigation: use them for experiments and snapshot-bound runs, while using HF Jobs or Modal when recurring reliability matters.
5. **Storage growth risk:** Append-only shards can grow beyond the `$0/month` target or `<$10/month` fallback, especially before compaction. Mitigation: set budget thresholds, review current/10x/100x growth, and schedule compaction or lifecycle cleanup as future implementation work.
6. **Manifest corruption risk:** The manifest is a small but critical control plane. Mitigation: publish manifests versioned and atomically, include checksums and parent links, keep previous known-good manifest versions, and move `latest` only after validation in future tooling.
7. **Stale reader risk:** Batch jobs can start before a new manifest is published and finish after it changes. Mitigation: readers must pin one manifest version at startup and continue on that version for reproducible retries.
8. **Compaction cost risk:** Physical compaction can require large reads and writes, especially at 10x and 100x scale. Mitigation: prune logically first, compact on a schedule, and delay physical deletion until rollback, snapshots, and pinned readers no longer need old shards.
9. **Derived artifact rebuild and publication-size risk:** Materialized samples, retrieval indexes, caches, and checkpoints may be large and expensive to rebuild or publish. Mitigation: treat derived artifacts separately from source shards, publish them only when permissions and size limits are clear, and record the manifest version each artifact depends on.
10. **Future hourly or near-real-time needs:** The recommended design targets daily or weekly batch freshness, not hourly ingestion, near-real-time serving, or low-latency interactive queries. Mitigation: reevaluate if update cadence or serving requirements change instead of stretching this lowest-cost batch design beyond its assumptions.

## Conceptual migration path only

This is the conceptual migration direction implied by the decision. It is not an implementation plan, schema, conversion script, upload plan, or manifest format; every execution step remains future work.

1. **Inventory source categories:** Treat `data/raw_matches.db`, `data/howtowin.db`, `data/splits/`, `data/materialized_samples/plan_b`, `data/retrieval/plan_b_index.pt`, `data/retrieval/cache/`, and `data/model_checkpoints/` as distinct source/artifact categories rather than one monolithic database upload.
2. **Map raw corpus:** Map raw matches and timelines from `data/raw_matches.db` to compressed raw shards. These preserve replay/reparse capability and are not optimized as the primary training scan format.
3. **Map normalized analytics corpus:** Map normalized `games`, `players`, `frames`, `events`, and materialized statistics from `data/howtowin.db` to analytics shards selected by manifest metadata for batch model runs.
4. **Map split metadata:** Map `data/splits/` holdout and split files to versioned metadata artifacts pinned by the same manifest version as the analytics shards used for a run.
5. **Map sample cache:** Map `data/materialized_samples/plan_b` to derived sample cache artifacts that can be regenerated from analytics shards or versioned separately when regeneration cost is high.
6. **Map retrieval indexes:** Map `data/retrieval/plan_b_index.pt` and `data/retrieval/cache/` to derived retrieval artifacts. They should not be treated as live source-of-truth data because they are rebuildable from pinned model/corpus inputs.
7. **Map checkpoints:** Map `data/model_checkpoints/` to separate model artifacts, not corpus shards. Checkpoints should reference the manifest version, split metadata, and retrieval/sample artifacts they depend on where future tooling supports that.
8. **Introduce manifest pinning conceptually:** Future batch jobs should select a manifest version before reading shards so stale reader behavior, rollback, and reproducible reruns are well-defined.
9. **Publish snapshots separately:** Periodic Kaggle or Hugging Face Datasets releases, if allowed by redistribution checks, should be generated from known manifest versions as archives or public snapshots. They should not become the live mutable append/prune source of truth.
10. **Keep execution approval separate:** Provider setup, credentials, shard conversion, loader changes, benchmarks, uploads, and deletion workflows are not approved by this ADR. They require future scoped tasks with their own verification.

## Rejected alternatives

### Kaggle as primary live mutable storage

Rejected for this ADR. Kaggle's recommended role is snapshot/archive distribution plus optional Kaggle notebook-native experimentation, not primary live storage and not the mutable source of truth.

The primary-live-storage role fails the future workload because append and prune would be expressed as repeated dataset version operations rather than cheap object additions, manifest edits, and later compaction. That creates avoidable operational friction for daily or weekly freshness, stale-reader handling, rollback, and physical deletion semantics in the recommended shard/manifest model.

This rejection does not reject Kaggle entirely. It rejects only Kaggle as the authoritative mutable store for the live corpus. Periodic public archive snapshots remain in scope if redistribution, privacy, and Kaggle terms are verified before execution.

## Reevaluation triggers

Revisit this decision if any of these concrete triggers occur:

1. The active corpus exceeds the agreed cost threshold, leaves the `$0/month` target without a clear path back, or pushes the `<$10/month` fallback beyond what storage growth or egress can justify.
2. Update cadence changes from daily or weekly to hourly or near-real-time, because manifest churn, orchestration frequency, and reader consistency would become different problems.
3. A private-data requirement changes, such as needing stricter private storage, faster deletion guarantees, restricted geographic placement, or more formal access auditing.
4. Kaggle, Hugging Face, Modal, Cloudflare R2, Backblaze B2, or another chosen platform changes API behavior, ToS, pricing, rate limits, free-tier rules, storage limits, or quota enforcement in a way that weakens the chosen design.
5. Batch runtime, GPU/CPU availability, storage requests, or network transfer exceeds free or cheap quotas often enough that the recommended stateless compute path is no longer the lowest-cost reliable option.
6. Filtered scans require low-latency indexes, predicate pushdown, or interactive serving rather than batch file-level shard fetch.
7. Pruning semantics require physical deletion faster than scheduled compaction can safely provide, especially for privacy, legal, or redistribution obligations.
8. Derived sample caches, retrieval indexes, checkpoints, or public snapshot bundles become too large to rebuild, publish, or keep in sync with pinned manifest versions under the current layout.

## Evidence appendix

Task 1 evidence lives in `.sisyphus/evidence/task-1-current-state.md` and `.sisyphus/evidence/task-1-scope-guardrail.md`.

Task 3 Kaggle source list:

1. Kaggle API documentation: `https://www.kaggle.com/docs/api`
2. Kaggle API README: `https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md`
3. Kaggle API OAuth-related pull request: `https://github.com/Kaggle/kaggle-api/pull/564`
4. Kaggle API dataset-version/download-related pull request: `https://github.com/Kaggle/kaggle-api/pull/335`

Task 3 evidence lives in `.sisyphus/evidence/task-3-kaggle-role.md` and `.sisyphus/evidence/task-3-kaggle-uncertainty.md`. Unknown numeric Kaggle quotas and unconfirmed partial streaming support are marked as verify-before-execution risks, not facts.

Task 4 source list:

1. Hugging Face Datasets streaming: `https://huggingface.co/docs/hub/en/datasets-streaming`
2. Hugging Face Storage Buckets: `https://huggingface.co/docs/hub/storage-buckets`
3. Hugging Face storage limits: `https://huggingface.co/docs/hub/storage-limits`
4. Hugging Face Jobs pricing: `https://huggingface.co/docs/hub/en/jobs-pricing`
5. Hugging Face Spaces storage: `https://huggingface.co/docs/hub/en/spaces-storage`
6. Modal pricing: `https://modal.com/pricing`
7. Modal Volumes: `https://modal.com/docs/guide/volumes`
8. Modal cloud bucket mounts: `https://modal.com/docs/guide/cloud-bucket-mounts`
9. Google Colab overview: `https://developers.google.com/colab`
10. Colab pricing: `https://colab.research.google.com/signup/pricing`
11. Replicate prediction rate limits: `https://replicate.com/docs/topics/predictions/rate-limits`
12. Fly.io pricing: `https://fly.io/docs/about/pricing/`
13. Fly.io volumes: `https://fly.io/docs/volumes/overview/`
14. Render free tier: `https://render.com/docs/free.md`
15. Render one-off jobs: `https://render.com/docs/one-off-jobs.md`
16. Render workflows: `https://render.com/docs/workflows.md`

Task 4 evidence lives in `.sisyphus/evidence/task-4-platform-coverage.md` and `.sisyphus/evidence/task-4-storage-compute-separation.md`.

Task 6 evidence lives in `.sisyphus/evidence/task-6-decision.md` and `.sisyphus/evidence/task-6-reevaluation.md`.
