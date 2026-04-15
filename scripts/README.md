# scripts/

Operational helpers. Not the app — just glue.

## sync-gpu.sh

Bidirectional rsync between this laptop and the WSL GPU host over meshnet/tailnet.

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
