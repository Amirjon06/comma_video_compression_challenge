"""Overfit the decoder to this one video, against the judges rather than pixels.

The pixel term is kept small but non-zero. Without it the model drifts toward
images that satisfy both judges without resembling the road, which is the
failure mode the whole approach is meant to avoid.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import frames as frames_mod
from . import judges
from .config import CACHE_DIR, NUM_PAIRS, POSE_DIMS
from .model import HNeRV


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-seg", type=float, default=1.0)
    ap.add_argument("--w-pose", type=float, default=1.0, help="multiplier on the score-matched pose weight")
    ap.add_argument("--w-pix", type=float, default=2e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-pairs", type=int, default=64)
    ap.add_argument("--out", type=Path, default=CACHE_DIR / "hnerv.pt")
    ap.add_argument("--resume", type=Path)
    return ap.parse_args()


def pose_weight(pose_distortion: float) -> float:
    """Match the pose term's pull to its actual influence on the score.

    score = 100*seg + sqrt(10*pose), so d(score)/d(pose) is sqrt(10)/(2*sqrt(pose))
    while d(score)/d(seg) is a constant 100. A fixed weight is therefore correct
    at exactly one point in training. This tracks the ratio instead, which
    matters because pose gets more sensitive the smaller it gets.
    """
    safe = max(pose_distortion, 1e-6)
    return float(math.sqrt(10.0) / (2.0 * math.sqrt(safe)) / 100.0)


@torch.no_grad()
def measure(model, net, small, seg_t, pose_t, device, num_pairs, seed=0):
    """Exact SegNet/PoseNet distortion on a fixed subset of pairs."""
    model.eval()
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(NUM_PAIRS, size=min(num_pairs, NUM_PAIRS), replace=False))
    seg_total, pose_total = 0.0, 0.0

    for start in range(0, len(ids), 4):
        chunk = ids[start : start + 4]
        idx = torch.from_numpy(chunk).to(device)
        recon = model.render_pairs(idx)
        pose_out, seg_out = judges.judge_outputs(net, recon)

        seg_ref = torch.from_numpy(np.asarray(seg_t[chunk])).to(device).float()
        pose_ref = torch.from_numpy(np.asarray(pose_t[chunk])).to(device).float()
        seg_total += (seg_out.argmax(1) != seg_ref.argmax(1)).float().mean((1, 2)).sum().item()
        pose_total += (
            (pose_out["pose"][..., :POSE_DIMS] - pose_ref).pow(2).mean(1).sum().item()
        )

    model.train()
    n = len(ids)
    return seg_total / n, pose_total / n


def main():
    args = parse_args()
    device = torch.device(args.device)

    small = frames_mod.load_small()
    seg_t, pose_t = judges.load_targets()
    net = judges.load_distortion_net(device)

    model = HNeRV().to(device)
    if args.resume and args.resume.exists():
        model.load_state_dict(torch.load(args.resume, map_location=device))
    model.train()

    report = model.param_report()
    print(json.dumps(report, indent=2))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    rng = np.random.default_rng(1234)
    started = time.time()
    best = math.inf
    w_pose = args.w_pose * pose_weight(1.0)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for step in range(1, args.steps + 1):
        pairs = rng.integers(0, NUM_PAIRS, size=args.batch_size)
        idx = torch.from_numpy(pairs).to(device)

        seg_ref = torch.from_numpy(np.asarray(seg_t[pairs])).to(device).float()
        pose_ref = torch.from_numpy(np.asarray(pose_t[pairs])).to(device).float()
        flat_ids = np.stack([2 * pairs, 2 * pairs + 1], 1).reshape(-1)
        target = torch.from_numpy(np.asarray(small[flat_ids])).to(device).float()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            recon = model.render_pairs(idx)
            pose_out, seg_out = judges.judge_outputs(net, recon)
            loss, seg_term, pose_term = judges.surrogate_loss(
                pose_out, seg_out, pose_ref, seg_ref, args.w_seg, w_pose
            )
            pix = F.mse_loss(recon.reshape(target.shape), target)
            loss = loss + args.w_pix * pix

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % 100 == 0:
            rate = step / (time.time() - started)
            print(
                f"step {step:6d}  loss {loss.item():.4f}  seg_ce {seg_term.item():.4f}  "
                f"pose {pose_term.item():.5f}  pix {pix.item():.1f}  "
                f"w_pose {w_pose:.4g}  {rate:.1f} it/s",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            seg_d, pose_d = measure(model, net, small, seg_t, pose_t, device, args.eval_pairs)
            partial = 100 * seg_d + math.sqrt(10 * pose_d)
            w_pose = args.w_pose * pose_weight(pose_d)
            print(
                f"[eval {step}] seg {seg_d:.6f} pose {pose_d:.6f} "
                f"distortion terms {partial:.4f}",
                flush=True,
            )
            if partial < best:
                best = partial
                args.out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.out)
                print(f"saved {args.out} at {partial:.4f}", flush=True)


if __name__ == "__main__":
    main()
