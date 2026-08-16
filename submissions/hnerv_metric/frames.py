"""Decode 0.mkv into a uint8 memmap using the repo's own decoder.

Going through AVVideoDataset rather than a raw ffmpeg pipe matters: the
evaluator compares against these exact pixels, and the repo's YUV->RGB uses
bilinear chroma upsampling with BT.601 limited range. A different conversion
would shift every target by a couple of levels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .config import (
    CACHE_DIR,
    CAMERA_H,
    CAMERA_W,
    NUM_FRAMES,
    REPO_ROOT,
    RENDER_H,
    RENDER_W,
)

sys.path.insert(0, str(REPO_ROOT))

FRAMES_PATH = CACHE_DIR / "frames_u8.npy"
SMALL_FRAMES_PATH = CACHE_DIR / "frames_small_u8.npy"


def export(video_dir: Path, out_path: Path) -> Path:
    from frame_utils import AVVideoDataset

    out_path.parent.mkdir(parents=True, exist_ok=True)
    store = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.uint8,
        shape=(NUM_FRAMES, CAMERA_H, CAMERA_W, 3),
    )

    ds = AVVideoDataset(
        ["0.mkv"],
        data_dir=video_dir,
        batch_size=8,
        device=torch.device("cpu"),
    )
    ds.prepare_data()

    written = 0
    for _, _, batch in ds:  # (B, 2, H, W, 3)
        flat = batch.reshape(-1, CAMERA_H, CAMERA_W, 3).numpy()
        take = min(flat.shape[0], NUM_FRAMES - written)
        store[written : written + take] = flat[:take]
        written += take
        print(f"\r{written}/{NUM_FRAMES} frames", end="", flush=True)
        if written >= NUM_FRAMES:
            break
    print()

    if written != NUM_FRAMES:
        raise RuntimeError(f"expected {NUM_FRAMES} frames, decoded {written}")

    store.flush()
    return out_path


def export_small(src_path: Path = FRAMES_PATH, out_path: Path = SMALL_FRAMES_PATH) -> Path:
    """Render-resolution copy, used as the pixel term during training."""
    import torch.nn.functional as F

    src = load(src_path)
    store = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8, shape=(NUM_FRAMES, 3, RENDER_H, RENDER_W)
    )
    for start in range(0, NUM_FRAMES, 16):
        end = min(start + 16, NUM_FRAMES)
        x = torch.from_numpy(np.asarray(src[start:end])).float().permute(0, 3, 1, 2)
        small = F.interpolate(x, size=(RENDER_H, RENDER_W), mode="bilinear")
        store[start:end] = small.round().clamp(0, 255).to(torch.uint8).numpy()
    store.flush()
    return out_path


def load(path: Path = FRAMES_PATH) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run: python -m submissions.hnerv_metric.frames")
    return np.load(path, mmap_mode="r")


def load_small(path: Path = SMALL_FRAMES_PATH) -> np.ndarray:
    return load(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, default=REPO_ROOT / "videos")
    ap.add_argument("--out", type=Path, default=FRAMES_PATH)
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()

    if not args.skip_full:
        path = export(args.video_dir, args.out)
        print(f"wrote {path} ({path.stat().st_size / 1e9:.2f} GB)")

    small = export_small(args.out)
    print(f"wrote {small} ({small.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
