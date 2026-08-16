"""Shared constants for the hnerv_metric submission."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".hnerv_cache"

# Native video geometry.
CAMERA_W, CAMERA_H = 1164, 874
NUM_FRAMES = 1200
NUM_PAIRS = NUM_FRAMES // 2

# Both judges bilinearly resize their input to this before doing anything,
# so nothing the model renders above it can affect the score.
JUDGE_W, JUDGE_H = 512, 384

# Model renders here, then bilinear up to camera size.
RENDER_W, RENDER_H = JUDGE_W, JUDGE_H

NUM_CLASSES = 5
POSE_DIMS = 6  # compute_distortion only uses the first half of the 12 outputs
