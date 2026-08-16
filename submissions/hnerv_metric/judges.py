"""Access to SegNet and PoseNet, both as the exact metric and as training losses.

The evaluator's distortions are not differentiable: SegNet's is an argmax
disagreement count, PoseNet's is an MSE on outputs. The MSE is usable as-is;
the argmax needs a surrogate, so training uses soft cross-entropy against the
original frame's logits and we check it still tracks the real number.

Targets for the original video never change, so they get computed once and
cached. Training then only pays for a forward pass on the reconstruction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    CACHE_DIR,
    CAMERA_H,
    CAMERA_W,
    JUDGE_H,
    JUDGE_W,
    NUM_CLASSES,
    NUM_PAIRS,
    POSE_DIMS,
    REPO_ROOT,
)

sys.path.insert(0, str(REPO_ROOT))

SEG_TARGET_PATH = CACHE_DIR / "seg_logits_f16.npy"
POSE_TARGET_PATH = CACHE_DIR / "pose_target_f32.npy"


def load_distortion_net(device: torch.device):
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def to_judge_input(frames: torch.Tensor) -> torch.Tensor:
    """(B, 2, 3, h, w) float in [0,255] -> the 512x384 tensor both judges see.

    The evaluator hands the judges native-resolution frames and they resize
    down. If the model already renders at judge resolution we skip the
    round trip, which is the same tensor up to the resampling the evaluator
    would have done to our own upsample.
    """
    b, t, c, h, w = frames.shape
    x = frames.reshape(b * t, c, h, w)
    if (h, w) != (JUDGE_H, JUDGE_W):
        x = F.interpolate(x, size=(JUDGE_H, JUDGE_W), mode="bilinear")
    return x.reshape(b, t, c, JUDGE_H, JUDGE_W)


def judge_outputs(net, judge_in: torch.Tensor):
    """judge_in: (B, 2, 3, 384, 512) float [0,255]. Returns pose dict, seg logits."""
    from frame_utils import rgb_to_yuv6

    b, t = judge_in.shape[:2]
    flat = judge_in.reshape(b * t, 3, JUDGE_H, JUDGE_W)
    yuv = rgb_to_yuv6(flat).reshape(b, t * 6, JUDGE_H // 2, JUDGE_W // 2)
    pose = net.posenet(yuv)
    seg = net.segnet(judge_in[:, -1])
    return pose, seg


def exact_distortion(pose_a, seg_a, pose_b, seg_b):
    """The evaluator's own two numbers, per sample."""
    pose = (
        (pose_a["pose"][..., :POSE_DIMS] - pose_b["pose"][..., :POSE_DIMS])
        .pow(2)
        .mean(dim=1)
    )
    seg = (seg_a.argmax(dim=1) != seg_b.argmax(dim=1)).float().mean(dim=(1, 2))
    return pose, seg


def surrogate_loss(pose_out, seg_out, pose_target, seg_target, w_seg=1.0, w_pose=1.0):
    """Differentiable stand-in for the score's distortion terms."""
    pose_term = (pose_out["pose"][..., :POSE_DIMS] - pose_target).pow(2).mean()
    target_prob = F.softmax(seg_target.float(), dim=1)
    seg_term = -(target_prob * F.log_softmax(seg_out, dim=1)).sum(dim=1).mean()
    return w_seg * seg_term + w_pose * pose_term, seg_term.detach(), pose_term.detach()


@torch.no_grad()
def build_targets(device: torch.device, batch_size: int = 4) -> None:
    """Cache SegNet logits and PoseNet outputs for the original video."""
    from . import frames as frames_mod

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    src = frames_mod.load()
    net = load_distortion_net(device)

    seg_store = np.lib.format.open_memmap(
        SEG_TARGET_PATH,
        mode="w+",
        dtype=np.float16,
        shape=(NUM_PAIRS, NUM_CLASSES, JUDGE_H, JUDGE_W),
    )
    pose_store = np.lib.format.open_memmap(
        POSE_TARGET_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(NUM_PAIRS, POSE_DIMS),
    )

    for start in range(0, NUM_PAIRS, batch_size):
        end = min(start + batch_size, NUM_PAIRS)
        chunk = np.asarray(src[2 * start : 2 * end])  # (2n, H, W, 3)
        x = torch.from_numpy(chunk).to(device).float()
        x = x.permute(0, 3, 1, 2).reshape(end - start, 2, 3, CAMERA_H, CAMERA_W)
        pose, seg = judge_outputs(net, to_judge_input(x))
        seg_store[start:end] = seg.to(torch.float16).cpu().numpy()
        pose_store[start:end] = pose["pose"][..., :POSE_DIMS].float().cpu().numpy()
        print(f"\rtargets {end}/{NUM_PAIRS}", end="", flush=True)
    print()

    seg_store.flush()
    pose_store.flush()


def load_targets():
    if not SEG_TARGET_PATH.exists():
        raise FileNotFoundError(
            "judge targets missing, run: python -m submissions.hnerv_metric.judges"
        )
    return np.load(SEG_TARGET_PATH, mmap_mode="r"), np.load(POSE_TARGET_PATH, mmap_mode="r")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    build_targets(torch.device(args.device), args.batch_size)
    print(f"wrote {SEG_TARGET_PATH}")
    print(f"wrote {POSE_TARGET_PATH}")


if __name__ == "__main__":
    main()
