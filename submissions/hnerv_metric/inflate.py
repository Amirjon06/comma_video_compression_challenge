"""Rebuild the raw frames from the archive payload."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .codec import decode_model
from .config import CAMERA_H, CAMERA_W, NUM_FRAMES
from .model import HNeRV


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def inflate(payload: Path, out_path: Path, device: torch.device, batch: int = 8) -> None:
    state = decode_model(payload.read_bytes())
    model = HNeRV()
    model.load_state_dict(state)
    model.eval().to(device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    store = np.memmap(
        out_path, mode="w+", dtype=np.uint8, shape=(NUM_FRAMES, CAMERA_H, CAMERA_W, 3)
    )

    started = time.time()
    with torch.inference_mode():
        for start in range(0, NUM_FRAMES, batch):
            end = min(start + batch, NUM_FRAMES)
            idx = torch.arange(start, end, device=device)
            small = model(idx)
            full = F.interpolate(small, size=(CAMERA_H, CAMERA_W), mode="bilinear")
            frames = full.clamp(0, 255).round().to(torch.uint8)
            store[start:end] = frames.permute(0, 2, 3, 1).cpu().numpy()
            if end % 200 == 0 or end == NUM_FRAMES:
                print(f"{end}/{NUM_FRAMES} frames in {time.time() - started:.1f}s", flush=True)

    store.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive_dir", type=Path)
    ap.add_argument("out_path", type=Path)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    payload = args.archive_dir / "p"
    if not payload.is_file():
        raise FileNotFoundError(payload)
    inflate(payload, args.out_path, pick_device(args.device), args.batch)


if __name__ == "__main__":
    main()
