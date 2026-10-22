# scripts/

Operational helpers. Not the app — just glue.

There are two GPU paths now:

- Use `sync-gpu.sh` for the WSL GPU host rsync workflow. It moves local DB snapshots and split files to the host, then pulls checkpoints back.
- Use the Kaggle workflow for a private, version-pinned Dataset mounted into a Kaggle Notebook. Kaggle is a separate free GPU snapshot path, not a replacement for `sync-gpu.sh` and not the future live mutable corpus or 100x data lake.

See `docs/kaggle_training_workflow.md` for the Kaggle notebook runbook. Keep using this file for local-to-WSL operations.

## sync-gpu.sh

Bidirectional rsync between this laptop and the WSL GPU host over meshnet/tailnet.

This path is separate from Kaggle. It can support mutable host-side iteration because it copies files over SSH. Kaggle should instead read a private dataset snapshot from `/kaggle/input` and write generated outputs only under `/kaggle/working/howtowin`.

### One-time setup (GPU host, inside WSL)

```bash
sudo apt update && sudo apt install -y openssh-server
sudo service ssh start
# persist across WSL restarts
sudo tee -a /etc/wsl.conf <<'EOF'
[boot]
command = "service ssh start"
EOF
```

### One-time setup (this laptop)

```bash
# generate a key if you don't have one
ls ~/.ssh/id_ed25519 2>/dev/null || ssh-keygen -t ed25519

# copy to the GPU host
ssh-copy-id <user>@xyn-himalayas.nord

# cache the target
echo 'export HOWL_GPU_HOST=<user>@xyn-himalayas.nord' >> ~/.zshrc
source ~/.zshrc
```

### Everyday use

```bash
scripts/sync-gpu.sh push    # corpus DBs + holdout split → GPU
scripts/sync-gpu.sh pull    # checkpoints ← GPU
```

### Driving a training run remotely

```bash
ssh "$HOWL_GPU_HOST" 'cd ~/build/howtowin.lol && git pull && cd code && \
  uv run python -m model.cli train && uv run python -m model.cli eval'
scripts/sync-gpu.sh pull
```

Wrap it in `tmux` on the GPU side for any run longer than a minute so an SSH disconnect doesn't kill the job.

## Kaggle snapshot workflow

Use this path when you want a free Kaggle GPU run against the current SQLite corpus snapshot. Do not use it for a live mutable corpus or for the future 100x data lake. Do not run actual upload commands from automation unless an operator explicitly chooses to do so.

Manual operator outline:

1. Prepare an upload-ready private dataset snapshot staging directory without uploading:

   ```bash
   uv run python scripts/prepare-kaggle-dataset.py \
     --source-root data \
     --staging-dir /tmp/howtowin-kaggle-staging
   ```

2. Inspect `/tmp/howtowin-kaggle-staging/data/howtowin.db`, `/tmp/howtowin-kaggle-staging/data/raw_matches.db`, `/tmp/howtowin-kaggle-staging/data/splits/`, `manifest.md`, `manifest.json`, `checksums.txt`, and `README.md`. Confirm only the DBs, split files, and metadata are in scope. Generated checkpoints, logs, reports, retrieval caches, and materialized samples should stay out. The default payload strategy avoids duplicating large DB files by using hardlinks where possible and symlinks as a fallback; use `--payload-strategy copy` only when an operator intentionally wants copies.

3. If the manifest is correct, run Kaggle CLI commands manually from a machine with your own Kaggle credentials configured outside this repo. Keep the slug placeholder-only in docs and scripts:

   ```bash
   kaggle datasets init -p /tmp/howtowin-kaggle-staging
   kaggle datasets create -p /tmp/howtowin-kaggle-staging --private
   kaggle datasets version -p /tmp/howtowin-kaggle-staging -m "refresh current training snapshot"
   ```

4. Pin the private dataset slug and version in the Kaggle Notebook. Do not rely silently on `latest`.

5. Follow `docs/kaggle_training_workflow.md` for the notebook environment cell, smoke command, training preflight, and `/kaggle/working` cleanup rules.

Security rules for Kaggle work:

- Do not commit `kaggle.json`, `KAGGLE_KEY`, OAuth material, notebook secrets, or tokens.
- Use placeholders such as `<dataset-slug>` and `<dataset-version>` in docs.
- Keep checkpoints, caches, and reports saved under `/kaggle/working/howtowin` below the 20 GB saved output cap. Delete unneeded files before saving a notebook version.

## upload_checkpoint.py

Prepare and optionally push curated checkpoints to a Hugging Face **model repo** using Git + LFS. This is the path to use when a checkpoint should have commit history / tags / branches; use plain `hf upload` for one-off blobs instead.

```bash
read -rsp 'HF token: ' HF_TOKEN && export HF_TOKEN && echo
export HOWL_HF_MODEL_REPO=nyx-haile/howtowin-plan-b-rssm-v1

python3 scripts/upload_checkpoint.py \
  --repo "$HOWL_HF_MODEL_REPO" \
  --dry-run \
  data/model_checkpoints/plan_b_full_best.pt

HF_TOKEN="$HF_TOKEN" python3 scripts/upload_checkpoint.py \
  --repo "$HOWL_HF_MODEL_REPO" \
  --push \
  data/model_checkpoints/plan_b_full_best.pt
```

See `docs/huggingface-model-repo-workflow.md` for the decision guide (`hf upload` vs Git + LFS), security notes, and the manual fallback commands.
