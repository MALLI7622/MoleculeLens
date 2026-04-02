# MoleculeSTM — GCP Cloud Setup Guide

A complete guide to setting up Google Cloud Platform (GCP) for training diffusion/generative models using JAX/Flax on TPUs and GPUs.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Architecture Overview](#architecture-overview)
4. [GCP Setup](#gcp-setup)
5. [Data Storage (GCS)](#data-storage-gcs)
6. [GPU VM Setup](#gpu-vm-setup)
7. [TPU VM Setup](#tpu-vm-setup)
8. [VS Code Remote Setup](#vs-code-remote-setup)
9. [Daily Workflow](#daily-workflow)
10. [Cost Overview](#cost-overview)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project uses Google Cloud TPUs/GPUs for training diffusion and generative models on molecular data. The setup is optimized for:

- **Framework**: JAX / Flax
- **Hardware**: NVIDIA L4 GPU (testing) + TPU v2-8 (full training)
- **Storage**: Google Cloud Storage (GCS)
- **OS**: Ubuntu 24.04 with CUDA 12.8

---

## Prerequisites

- macOS local machine
- Google Cloud account with active credits
- GCP Project ID: `interpretable-ml-moleculelens`
- TPU Builders Program access
- VS Code installed ([code.visualstudio.com](https://code.visualstudio.com))

---

## Architecture Overview

```
Local Mac
│
├── gcloud CLI          → manages VMs, TPUs, GCS
├── VS Code             → remote editing via SSH
│
└── Google Cloud
    ├── GCS Bucket (gs://molecule-lens/)   → dataset + checkpoints (persistent)
    ├── GPU VM (g2-standard-4 + L4)        → prototyping & testing
    └── TPU VM (v2-8)                      → full training runs
```

---

## GCP Setup

### 1. Install gcloud CLI

```bash
# Install
curl https://sdk.cloud.google.com | bash

# Restart shell
exec -l $SHELL

# Verify
gcloud --version
```

### 2. Authenticate & Set Project

```bash
gcloud init
# Follow prompts — sign in and select project: interpretable-ml-moleculelens

# Verify
gcloud config get-value project
```

### 3. Request GPU Quota

By default GCP projects have 0 GPU quota. Request it:

1. Go to [console.cloud.google.com/iam-admin/quotas](https://console.cloud.google.com/iam-admin/quotas)
2. Search `GPUS_ALL_REGIONS`
3. Click **Edit Quota** → set to `1`
4. Submit with reason: *"ML research - diffusion model training"*
5. Approval is usually instant via email from `cloudquota@google.com`

### 4. Set Up Billing Alerts

1. Go to [console.cloud.google.com/billing](https://console.cloud.google.com/billing)
2. Click **Budgets & Alerts** → **Create Budget**
3. Set amount and alert thresholds at 50%, 90%, 100%

---

## Data Storage (GCS)

> 💡 Always store datasets and checkpoints in GCS — not on the VM disk. VMs can be deleted, GCS persists.

### Create Bucket

```bash
# Bucket names must be lowercase, no underscores
gsutil mb -l us-central1 gs://molecule-lens/
```

### Upload Dataset from Local Machine

```bash
# Upload with parallel flag for speed
gsutil -m cp -r workspace.tar.gz gs://molecule-lens/

# Verify upload
gsutil ls -l gs://molecule-lens/

# Check total size
gsutil du -sh gs://molecule-lens/
```

### Download Dataset to VM

```bash
# Run inside the VM after SSH
gsutil cp gs://molecule-lens/workspace.tar.gz .
tar -xzf workspace.tar.gz
```

> ✅ Data transfer within `us-central1` is **free**.

---

## GPU VM Setup

Use the GPU VM for **prototyping and testing** before committing to expensive TPU runs.

### Find Available Images

```bash
gcloud compute images list \
  --project=deeplearning-platform-release \
  --filter="family~'common-cu'" \
  --format="table(family, name)" \
  --no-standard-images | head -20
```

### Create GPU VM

```bash
gcloud compute instances create my-gpu-vm \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --image-family=common-cu128-ubuntu-2404-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB
```

> ⚠️ If you get `ZONE_RESOURCE_POOL_EXHAUSTED`, try `us-central1-b` or `us-central1-c`.

**VM Specs:**

| Component | Details |
|-----------|---------|
| GPU | NVIDIA L4 (24GB VRAM) |
| vCPU | 4 cores |
| RAM | 16GB |
| Disk | 200GB SSD |
| CUDA | 12.8 |

### SSH Into GPU VM

```bash
gcloud compute ssh my-gpu-vm --zone=us-central1-a
```

> 💡 Wait 30-60 seconds after starting the VM before SSH — it needs time to boot.

### Verify GPU

```bash
nvidia-smi
# Should show NVIDIA L4 with 23034MiB memory
```

### Set Up Python Environment (inside VM)

```bash
# Start tmux to prevent losing work on disconnect
tmux new -s main

# Activate conda (if not loaded automatically)
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create and activate environment
conda create -n moleculestm python=3.10 -y
conda activate moleculestm

# Install JAX with GPU support
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install project dependencies
pip install flax optax diffusers transformers datasets
pip install orbax-checkpoint wandb tqdm pillow
```

---

## TPU VM Setup

Use the TPU VM for **full training runs** only. TPU VMs cannot be stopped — only deleted.

### Check Available TPU Types

```bash
gcloud compute tpus accelerator-types list --zone=us-central1-f
# Output: v2-8
```

### Create TPU VM

```bash
gcloud compute tpus tpu-vm create my-tpu-vm \
  --zone=us-central1-f \
  --accelerator-type=v2-8 \
  --version=tpu-ubuntu2204-base
```

### SSH Into TPU VM

```bash
gcloud compute tpus tpu-vm ssh my-tpu-vm --zone=us-central1-f
```

### Install JAX for TPU (inside VM)

```bash
# CRITICAL: Install TPU-specific JAX, not CPU/GPU version
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Verify TPU detected
python3 -c "import jax; print(jax.devices())"
# Should show 8 TPU devices
```

### Delete TPU VM When Done

```bash
# Always delete when done — TPUs can't be stopped!
gcloud compute tpus tpu-vm delete my-tpu-vm --zone=us-central1-f
```

> ⚠️ Always push code to GitHub and save checkpoints to GCS before deleting.

---

## VS Code Remote Setup

### 1. Install Extension

In VS Code: `Cmd+Shift+X` → search **"Remote - SSH"** → Install

### 2. Get VM External IP

```bash
gcloud compute instances describe my-gpu-vm \
  --zone=us-central1-a \
  --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

### 3. Configure SSH

Edit `~/.ssh/config` on your Mac:

```
Host gcp-gpu-vm
    HostName <EXTERNAL_IP>
    User cheriearjun
    IdentityFile ~/.ssh/google_compute_engine
```

> ⚠️ External IP changes every time the VM restarts — update this file each time.

### 4. Connect

`Cmd+Shift+P` → **Remote-SSH: Connect to Host** → `gcp-gpu-vm`

---

## Daily Workflow

### ▶️ Starting Work (from Mac terminal)

```bash
# 1. Start VM
gcloud compute instances start my-gpu-vm --zone=us-central1-a

# 2. Wait ~30 seconds, then SSH in
gcloud compute ssh my-gpu-vm --zone=us-central1-a

# 3. Inside VM — reattach tmux
tmux attach -t main
# or start new session
tmux new -s main

# 4. Activate environment
conda activate moleculestm
```

### ⏹️ Ending Work (from Mac terminal)

```bash
# ALWAYS stop from your Mac, not from inside the VM!
gcloud compute instances stop my-gpu-vm --zone=us-central1-a
```

### 🔁 If Disconnected

```bash
# SSH back in
gcloud compute ssh my-gpu-vm --zone=us-central1-a

# Reattach tmux session — your work is still running!
tmux attach -t main
```

---

## Cost Overview

| Resource | Running Cost | Stopped Cost |
|----------|-------------|--------------|
| GPU VM (L4 g2-standard-4) | ~$0.77/hr | ~$8/month (disk) |
| TPU VM (v2-8) | ~$12/hr | N/A (can't stop) |
| GCS Storage (per 50GB) | — | ~$1/month |
| Data transfer (same region) | Free | Free |

### Example Daily Cost (GPU VM)

```
4 hrs working  →  4 × $0.77 = $3.08
20 hrs stopped →  $0.20
Total/day      →  ~$3.28
```

### Cost Saving Tips

| Tip | Saving |
|-----|--------|
| Use GPU VM for testing, TPU only for full runs | ~10x savings |
| Use Kaggle free TPU (20hrs/month) for prototyping | 100% free |
| Always stop GPU VM when idle | Saves ~$0.77/hr |
| Keep checkpoints in GCS, delete TPU VM after run | Avoids idle TPU charges |
| Use `tmux` to avoid re-running jobs after disconnect | Saves time & compute |

---

## Troubleshooting

### SSH Connection Refused
```bash
# VM just started — wait 30-60 seconds and retry
sleep 30 && gcloud compute ssh my-gpu-vm --zone=us-central1-a
```

### conda Not Found After Restart
```bash
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Zone Resource Pool Exhausted (GPU)
```bash
# Try other zones in us-central1
--zone=us-central1-b
--zone=us-central1-c
```

### Bucket Not Found
```bash
# List all your buckets
gsutil ls
# Use the exact bucket name shown
```

### Stop Command Permission Denied
```bash
# You're inside the VM — exit first!
exit
# Then stop from your Mac terminal
gcloud compute instances stop my-gpu-vm --zone=us-central1-a
```

### VM Terminated Automatically
```bash
# Normal with --maintenance-policy=TERMINATE, just restart
gcloud compute instances start my-gpu-vm --zone=us-central1-a
```

---

## Project Structure (Recommended)

```
MoleculeSTM/
├── README.md                  ← this file
├── installation.md            ← detailed installation steps
├── setup.sh                   ← environment setup script
├── data/                      ← local data (gitignored)
├── checkpoints/               ← local checkpoints (gitignored)
├── src/
│   ├── models/                ← JAX/Flax model definitions
│   ├── training/              ← training scripts
│   └── utils/                 ← helper functions
└── experiments/               ← experiment configs & logs
```

### setup.sh (run after creating a new VM)

```bash
#!/bin/bash
# Reinstall environment on a fresh VM

conda create -n moleculestm python=3.10 -y
conda activate moleculestm

pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax diffusers transformers datasets
pip install orbax-checkpoint wandb tqdm pillow numpy

echo "✅ Environment setup complete!"
```

---

## Contributing

1. Fork the repository
2. Set up your GCP environment following this guide
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Push checkpoints to GCS before committing code
5. Submit a Pull Request

---

*Last updated: March 2026 | GCP Project: interpretable-ml-moleculelens*
