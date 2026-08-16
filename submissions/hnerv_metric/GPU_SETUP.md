# Running the training on a rented GPU

Nothing large gets uploaded. The GPU box clones the repo and rebuilds the
caches itself, which takes a few minutes there versus an hour on a laptop.

## Renting

vast.ai or RunPod. Pick an RTX 4090 or A100, 50 GB disk, a PyTorch/CUDA
template. Around $0.35/hr for a 4090. Open its terminal and work there.

Disk matters: the frame caches are about 5.5 GB.

## Setup on the box

Clone comma's repo rather than the fork. The video and the two judge networks
are Git LFS objects and GitHub does not always carry them into a fork, so
pulling LFS from upstream is the reliable path. The branch comes from the fork
afterwards.

```bash
apt-get update && apt-get install -y git-lfs ffmpeg
git lfs install

git clone https://github.com/commaai/comma_video_compression_challenge.git
cd comma_video_compression_challenge
git lfs pull

git remote add fork https://github.com/Amirjon06/comma_video_compression_challenge.git
git fetch fork hnerv-metric
git checkout -b hnerv-metric fork/hnerv-metric

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --group cu126
source .venv/bin/activate
```

Confirm the GPU is visible before spending time on anything else:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Build caches

```bash
python -m submissions.hnerv_metric.frames
python -m submissions.hnerv_metric.judges --device cuda
python -m submissions.hnerv_metric.selftest
```

## Train

Run it under `tmux` so a dropped SSH session does not kill the job.

```bash
tmux new -s train
python -m submissions.hnerv_metric.train \
  --device cuda --amp \
  --steps 60000 --batch-size 8 \
  --eval-every 2000 --eval-pairs 96 \
  2>&1 | tee train.log
```

Detach with `Ctrl+B` then `D`. Reattach with `tmux attach -t train`.

## Pack and check

```bash
bash submissions/hnerv_metric/compress.sh
bash evaluate.sh --submission-dir ./submissions/hnerv_metric --device cuda
cat submissions/hnerv_metric/report.txt
```

## Bring home

Only two files matter. Everything else is reproducible.

```bash
# from your laptop
scp -P <port> root@<host>:~/comma_video_compression_challenge/.hnerv_cache/hnerv.pt .
scp -P <port> root@<host>:~/comma_video_compression_challenge/train.log .
```

Destroy the instance when the copy finishes. Billing is by the minute and a
forgotten box is the only way this gets expensive.

## What to watch in the log

- `seg_ce` should keep falling. Flat early means the learning rate is wrong.
- `[eval]` `seg` is the real metric. Under 0.01 by step 20k is on track.
- `pose` should start dropping once the picture is coherent, usually a few
  thousand steps in. `w_pose` rises automatically as pose improves.
- `distortion terms` is the score without the rate term. Budget is about 0.10
  if the archive lands near 130 KB.
