"""Find out why the pose loss will not move.

Four checks, cheapest first:

1. Feed PoseNet the real downsampled frames. If pose distortion is not near
   zero, the cached targets and the training path disagree and the model was
   never the problem.
2. Feed the model's output and print the raw pose vectors. A constant output
   regardless of input means the gradient is dead at the source.
3. Measure gradient magnitude arriving at the rendered image from each term.
4. Feed the real frames with small perturbations to see how sensitive PoseNet
   actually is.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from . import frames as frames_mod
from . import judges
from .config import POSE_DIMS
from .model import HNeRV


def load_batch(small, pairs, device):
    ids = np.stack([2 * pairs, 2 * pairs + 1], 1).reshape(-1)
    arr = np.asarray(small[ids]).astype(np.float32)
    x = torch.from_numpy(arr).to(device)
    return x.reshape(len(pairs), 2, 3, x.shape[-2], x.shape[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=".hnerv_cache/probe.pt")
    ap.add_argument("--pairs", type=int, default=8)
    args = ap.parse_args()

    device = torch.device(args.device)
    small = frames_mod.load_small()
    seg_t, pose_t = judges.load_targets()
    net = judges.load_distortion_net(device)

    pairs = np.arange(args.pairs)
    pose_ref = torch.from_numpy(np.asarray(pose_t[pairs])).to(device).float()
    seg_ref = torch.from_numpy(np.asarray(seg_t[pairs])).to(device).float()

    print("=== 1. real frames through the training path ===")
    real = load_batch(small, pairs, device)
    with torch.no_grad():
        pose_real, seg_real = judges.judge_outputs(net, real)
    got = pose_real["pose"][..., :POSE_DIMS]
    mse_real = (got - pose_ref).pow(2).mean().item()
    seg_real_d = (seg_real.argmax(1) != seg_ref.argmax(1)).float().mean().item()
    print(f"pose MSE with true frames: {mse_real:.6f}")
    print(f"seg disagreement with true frames: {seg_real_d:.6f}")
    print("target pose row 0:", pose_ref[0].tolist())
    print("actual pose row 0:", got[0].tolist())

    print()
    print("=== 2. model output ===")
    model = HNeRV().to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"loaded {args.checkpoint}")
    except FileNotFoundError:
        print(f"{args.checkpoint} not found, using an untrained model")
    model.eval()

    with torch.no_grad():
        recon = model.render_pairs(torch.from_numpy(pairs).to(device))
        pose_recon, _ = judges.judge_outputs(net, recon)
    rec = pose_recon["pose"][..., :POSE_DIMS]
    print(f"pose MSE with model output: {(rec - pose_ref).pow(2).mean().item():.6f}")
    print("model pose row 0:", rec[0].tolist())
    print("model pose row 1:", rec[1].tolist())
    print(f"spread across the batch (std per dim): {rec.std(dim=0).tolist()}")
    print(f"target spread across the batch:        {pose_ref.std(dim=0).tolist()}")

    print()
    print("=== 3. gradient reaching the rendered image ===")
    recon = model.render_pairs(torch.from_numpy(pairs).to(device))
    recon.retain_grad()
    pose_out, seg_out = judges.judge_outputs(net, recon)
    pose_loss = (pose_out["pose"][..., :POSE_DIMS] - pose_ref).pow(2).mean()
    pose_loss.backward(retain_graph=True)
    pose_grad = recon.grad.abs().mean().item()

    recon2 = model.render_pairs(torch.from_numpy(pairs).to(device))
    recon2.retain_grad()
    _, seg_out2 = judges.judge_outputs(net, recon2)
    seg_loss, _, _ = judges.surrogate_loss(
        {"pose": torch.zeros(len(pairs), 12, device=device)},
        seg_out2,
        torch.zeros(len(pairs), POSE_DIMS, device=device),
        seg_ref,
        1.0,
        0.0,
    )
    seg_loss.backward()
    seg_grad = recon2.grad.abs().mean().item()

    print(f"mean |d pose_loss / d pixel|: {pose_grad:.3e}")
    print(f"mean |d seg_loss  / d pixel|: {seg_grad:.3e}")
    print(f"ratio seg/pose: {seg_grad / max(pose_grad, 1e-30):.1f}")

    print()
    print("=== 4. PoseNet sensitivity to real frames ===")
    for noise in (0.0, 1.0, 5.0, 20.0):
        noisy = (real + noise * torch.randn_like(real)).clamp(0, 255)
        with torch.no_grad():
            pose_n, _ = judges.judge_outputs(net, noisy)
        val = (pose_n["pose"][..., :POSE_DIMS] - pose_ref).pow(2).mean().item()
        print(f"noise sigma {noise:5.1f} levels -> pose MSE {val:.6f}")

    blur = torch.nn.functional.avg_pool2d(real.flatten(0, 1), 5, 1, 2).reshape(real.shape)
    with torch.no_grad():
        pose_b, _ = judges.judge_outputs(net, blur)
    print(f"5x5 blur -> pose MSE {(pose_b['pose'][..., :POSE_DIMS] - pose_ref).pow(2).mean().item():.6f}")

    swapped = real.flip(1)
    with torch.no_grad():
        pose_s, _ = judges.judge_outputs(net, swapped)
    print(f"frames swapped -> pose MSE {(pose_s['pose'][..., :POSE_DIMS] - pose_ref).pow(2).mean().item():.6f}")


if __name__ == "__main__":
    main()
