# Hugging Face model repo workflow (git + LFS)

Use this workflow when a checkpoint should live in a **versioned Hugging Face model repo** instead of being pushed as a one-off blob.

## Choose the right upload path

| Situation | Prefer | Why |
| --- | --- | --- |
| One-off artifact: eval report, retrieval index, single archival blob, or a file you do not plan to revise in place | `hf upload` | Fastest path; no separate clone, branch, or Git history to manage. |
| Named milestone checkpoints you want to browse, diff by filename, tag, roll back, or keep on branches | Git + LFS model repo | Treats weights like release artifacts with commit history and normal repo ergonomics. |
| Frequent training iterations where only a few checkpoints are worth keeping (`plan_b_smoke`, `plan_b_full_best`, `canonical_v1`) | Git + LFS model repo for the promoted milestones | Gives a stable lineage for the checkpoints that matter. |
| High-churn scratch snapshots or every-N-steps autosaves | Usually **not** Git + LFS | LFS history grows quickly; keep throwaways local or ship them to a bucket/object store instead. |

Rule of thumb: **`hf upload` ships blobs; Git + LFS manages checkpoint lineage.**

## Prereqs

- `git`
- `git-lfs` installed locally
  - Debian/Ubuntu: `sudo apt install -y git-lfs`
  - Arch: `sudo pacman -S git-lfs`
- Git identity configured (`git config --global user.name ...` and `git config --global user.email ...`)
- An existing Hugging Face **model** repo (create it in the web UI or with the CLI)
- A write token exported in the current shell as `HF_TOKEN`

If you need to create the model repo first, create it in the UI or with the CLI:

```bash
export HOWL_HF_MODEL_REPO=nyx-haile/howtowin-plan-b-rssm-v1
HF_TOKEN="$HF_TOKEN" hf repo create "$HOWL_HF_MODEL_REPO" --type model
```

## Authentication: env var only

Do **not** bake the token into the Git remote URL and do **not** persist it to `~/.git-credentials`.

Recommended pattern for one shell session:

```bash
read -rsp 'HF token: ' HF_TOKEN && export HF_TOKEN && echo
export HOWL_HF_MODEL_REPO=nyx-haile/howtowin-plan-b-rssm-v1
```

When done:

```bash
unset HF_TOKEN
```

## Recommended helper: `scripts/upload_checkpoint.py`

The helper keeps a separate local clone of the model repo under:

```text
~/.cache/howtowin/hf-model-repos/<owner>__<repo>
```

It then:

1. clones or updates that model repo clone,
2. runs `git lfs install --local --skip-smudge`,
3. tracks common checkpoint extensions (`*.pt`, `*.pth`, `*.ckpt`, `*.bin`, `*.safetensors`),
4. copies the requested artifacts into `checkpoints/` by default,
5. commits locally, and
6. optionally pushes with a temporary `GIT_ASKPASS` helper that reads `HF_TOKEN` from the environment.

That means **no token in the remote URL, no plaintext credential helper, and no secret committed to this repo**.

### Dry-run / preview

```bash
python3 scripts/upload_checkpoint.py \
  --repo "$HOWL_HF_MODEL_REPO" \
  --dry-run \
  data/model_checkpoints/plan_b_full_best.pt
```

### Prepare a local commit only

```bash
HF_TOKEN="$HF_TOKEN" python3 scripts/upload_checkpoint.py \
  --repo "$HOWL_HF_MODEL_REPO" \
  --commit-message "Add plan_b_full_best checkpoint" \
  data/model_checkpoints/plan_b_full_best.pt
```

### Commit and push

```bash
HF_TOKEN="$HF_TOKEN" python3 scripts/upload_checkpoint.py \
  --repo "$HOWL_HF_MODEL_REPO" \
  --push \
  data/model_checkpoints/plan_b_full_best.pt
```

### Useful flags

- `--dest-dir <path>` — place files somewhere other than `checkpoints/`
- `--worktree <dir>` — override the cached clone location
- `--track <glob>` — add extra `git-lfs` patterns
- `--skip-pull` — skip the pre-copy `git pull --ff-only`
- `--allow-dirty` — only if you intentionally want to reuse a dirty cached clone

## Manual fallback (no helper)

```bash
MODEL_REPO="$HOWL_HF_MODEL_REPO"
WORKTREE="${XDG_CACHE_HOME:-$HOME/.cache}/howtowin/hf-model-repos/${MODEL_REPO/\//__}"
ASKPASS="$(mktemp)"

cat > "$ASKPASS" <<'EOS'
#!/usr/bin/env bash
case "${1:-}" in
  *Username*|*username*) printf '%s\n' '__token__' ;;
  *Password*|*password*) printf '%s\n' "${HF_TOKEN:?set HF_TOKEN}" ;;
  *) exit 1 ;;
esac
EOS
chmod 700 "$ASKPASS"

mkdir -p "$(dirname "$WORKTREE")"
if [ ! -d "$WORKTREE/.git" ]; then
  GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="$ASKPASS" GIT_LFS_SKIP_SMUDGE=1 \
    git -c credential.helper= clone "https://huggingface.co/$MODEL_REPO" "$WORKTREE"
fi

cd "$WORKTREE"
git lfs install --local --skip-smudge
git lfs track '*.pt' '*.pth' '*.ckpt' '*.bin' '*.safetensors'
GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="$ASKPASS" \
  git -c credential.helper= pull --ff-only origin main || true
mkdir -p checkpoints
cp -f /absolute/path/to/plan_b_full_best.pt checkpoints/
git add .gitattributes checkpoints/plan_b_full_best.pt
git commit -m 'Add plan_b_full_best checkpoint'
GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="$ASKPASS" \
  git -c credential.helper= push origin HEAD

rm -f "$ASKPASS"
unset HF_TOKEN
```

If the remote repo already exists locally, skip the clone step and start from `git pull --ff-only origin main`.

## Suggested team convention

- Promote only **named** or **decision-relevant** checkpoints into the model repo.
- Keep scratch / high-frequency autosaves out of LFS unless they are part of a deliberate retention plan.
- Put lightweight reports or retrieval bundles on HF with `hf upload` unless they truly benefit from Git history next to the weights.
